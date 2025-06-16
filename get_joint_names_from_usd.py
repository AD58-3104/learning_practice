#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    from pxr import Usd, UsdPhysics, UsdGeom, Sdf
except ImportError:
    print("エラー: usd-coreがインストールされていません")
    sys.exit(1)

def find_all_joints(usd_file_path, verbose=False):
    """USDファイルからすべてのタイプのジョイントを探す"""
    
    stage = Usd.Stage.Open(str(usd_file_path))
    if not stage:
        print(f"エラー: ファイルを開けません: {usd_file_path}")
        return
    
    print(f"\nUSDファイル: {usd_file_path}")
    print("=" * 70)
    
    # 各種ジョイントをカウント
    joint_types = defaultdict(list)
    articulation_roots = []
    
    # すべてのプリムを調査
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        
        # Physics Joints をチェック
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joint_type = prim.GetTypeName()
            joint_types[joint_type].append(prim_path)
            
            if verbose:
                print(f"\n[Physics Joint発見]")
                print(f"  パス: {prim_path}")
                print(f"  タイプ: {joint_type}")
                analyze_physics_joint(joint, prim)
        
        # 特定のジョイントタイプをチェック
        elif prim.IsA(UsdPhysics.RevoluteJoint):
            joint_types["RevoluteJoint"].append(prim_path)
            if verbose:
                print(f"\n[Revolute Joint] {prim_path}")
                analyze_revolute_joint(prim)
                
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_types["PrismaticJoint"].append(prim_path)
            if verbose:
                print(f"\n[Prismatic Joint] {prim_path}")
                analyze_prismatic_joint(prim)
                
        elif prim.IsA(UsdPhysics.SphericalJoint):
            joint_types["SphericalJoint"].append(prim_path)
            if verbose:
                print(f"\n[Spherical Joint] {prim_path}")
                
        elif prim.IsA(UsdPhysics.FixedJoint):
            joint_types["FixedJoint"].append(prim_path)
            if verbose:
                print(f"\n[Fixed Joint] {prim_path}")
        
        # Articulation Root をチェック
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(prim_path)
            if verbose:
                print(f"\n[Articulation Root] {prim_path}")
        
        # プリム名に'joint'が含まれる場合も表示（念のため）
        if 'joint' in prim.GetName().lower() and not any(prim_path in joints for joints in joint_types.values()):
            if verbose:
                print(f"\n[名前にjointを含むプリム] {prim_path} (Type: {prim.GetTypeName()})")
    
    # 結果のサマリーを表示
    print("\n--- ジョイントのサマリー ---")
    total_joints = sum(len(joints) for joints in joint_types.values())
    
    if total_joints == 0:
        print("ジョイントが見つかりませんでした。")
        
        # 代替の調査
        print("\n--- 代替調査 ---")
        check_alternative_structures(stage)
    else:
        for joint_type, paths in joint_types.items():
            if paths:
                print(f"\n{joint_type}: {len(paths)}個")
                if not verbose:
                    # verbose=Falseの場合、最初の3つだけ表示
                    for i, path in enumerate(paths[:3]):
                        print(f"  - {path}")
                    if len(paths) > 3:
                        print(f"  ... 他 {len(paths) - 3} 個")
        
        print(f"\n合計ジョイント数: {total_joints}")
        
        if articulation_roots:
            print(f"\nArticulation Roots: {len(articulation_roots)}個")
            for root in articulation_roots:
                print(f"  - {root}")
    
    # ジョイント階層を表示
    if total_joints > 0 and verbose:
        print("\n--- ジョイント階層 ---")
        display_joint_hierarchy(stage, joint_types)

def analyze_physics_joint(joint, prim):
    """Physics Jointの詳細を分析"""
    # Body0とBody1の関係を取得
    body0_rel = joint.GetBody0Rel()
    body1_rel = joint.GetBody1Rel()
    
    if body0_rel.GetTargets():
        print(f"  Body0: {body0_rel.GetTargets()[0]}")
    if body1_rel.GetTargets():
        print(f"  Body1: {body1_rel.GetTargets()[0]}")
    
    # ローカル位置を取得
    local_pos0 = joint.GetLocalPos0Attr()
    local_pos1 = joint.GetLocalPos1Attr()
    
    if local_pos0.HasValue():
        print(f"  LocalPos0: {local_pos0.Get()}")
    if local_pos1.HasValue():
        print(f"  LocalPos1: {local_pos1.Get()}")

def analyze_revolute_joint(prim):
    """Revolute Joint (回転ジョイント) の詳細"""
    joint = UsdPhysics.RevoluteJoint(prim)
    
    # 軸を取得
    axis_attr = joint.GetAxisAttr()
    if axis_attr.HasValue():
        print(f"  回転軸: {axis_attr.Get()}")
    
    # 制限を取得
    lower_limit = joint.GetLowerLimitAttr()
    upper_limit = joint.GetUpperLimitAttr()
    
    if lower_limit.HasValue() and upper_limit.HasValue():
        print(f"  制限: [{lower_limit.Get():.2f}, {upper_limit.Get():.2f}] degrees")

def analyze_prismatic_joint(prim):
    """Prismatic Joint (直動ジョイント) の詳細"""
    joint = UsdPhysics.PrismaticJoint(prim)
    
    # 軸を取得
    axis_attr = joint.GetAxisAttr()
    if axis_attr.HasValue():
        print(f"  移動軸: {axis_attr.Get()}")
    
    # 制限を取得
    lower_limit = joint.GetLowerLimitAttr()
    upper_limit = joint.GetUpperLimitAttr()
    
    if lower_limit.HasValue() and upper_limit.HasValue():
        print(f"  制限: [{lower_limit.Get():.2f}, {upper_limit.Get():.2f}] units")

def check_alternative_structures(stage):
    """代替のジョイント構造をチェック"""
    
    # すべての関係性をチェック
    relationships = defaultdict(list)
    constraints = []
    
    for prim in stage.Traverse():
        # Constraint API をチェック
        if prim.HasAPI(UsdPhysics.MassAPI):
            print(f"  MassAPI: {prim.GetPath()}")
        
        # 関係性を持つプリムをチェック
        for rel in prim.GetRelationships():
            if rel.GetTargets():
                relationships[rel.GetName()].append(str(prim.GetPath()))
        
        # プリムの属性から手がかりを探す
        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            if any(keyword in attr_name.lower() for keyword in ['joint', 'link', 'constraint', 'connect']):
                print(f"  関連属性: {prim.GetPath()}.{attr_name}")

def display_joint_hierarchy(stage, joint_types):
    """ジョイントの階層構造を表示"""
    
    # すべてのジョイントパスを収集
    all_joint_paths = []
    for paths in joint_types.values():
        all_joint_paths.extend(paths)
    
    # 親子関係を構築
    parent_child = defaultdict(list)
    joint_prims = {}
    
    for joint_path in all_joint_paths:
        prim = stage.GetPrimAtPath(joint_path)
        if prim:
            joint_prims[joint_path] = prim
            
            # Physics Jointの場合、Body0とBody1から階層を推定
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                body1_targets = joint.GetBody1Rel().GetTargets()
                if body1_targets:
                    child_path = str(body1_targets[0])
                    parent_child[joint_path].append(child_path)
    
    # ルートジョイントから表示
    displayed = set()
    
    def print_hierarchy(path, indent=0):
        if path in displayed:
            return
        displayed.add(path)
        
        prefix = "  " * indent + "└─ "
        if path in joint_prims:
            prim = joint_prims[path]
            print(f"{prefix}{prim.GetName()} ({prim.GetTypeName()})")
        else:
            print(f"{prefix}{Path(path).name}")
        
        for child in parent_child.get(path, []):
            print_hierarchy(child, indent + 1)
    
    # ルートから開始
    for joint_path in all_joint_paths:
        if joint_path not in displayed:
            print_hierarchy(joint_path)

def main():
    parser = argparse.ArgumentParser(
        description='USDファイルからPhysics Jointsを含むすべてのジョイント情報を抽出',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('usd_file', help='読み込むUSDファイル')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='詳細な情報を表示')
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    if not Path(args.usd_file).exists():
        print(f"エラー: ファイルが見つかりません: {args.usd_file}")
        sys.exit(1)
    
    find_all_joints(args.usd_file, args.verbose)

if __name__ == "__main__":
    main()