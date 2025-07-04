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

def find_all_joints(usd_file_path, verbose=False, joint_details=False):
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
            
            if joint_details:
                print(f"\n[Physics Joint詳細情報]")
                print(f"  パス: {prim_path}")
                print(f"  タイプ: {joint_type}")
                analyze_physics_joint_details(joint, prim)
        
        # 特定のジョイントタイプをチェック
        elif prim.IsA(UsdPhysics.RevoluteJoint):
            joint_types["RevoluteJoint"].append(prim_path)
            if verbose:
                print(f"\n[Revolute Joint] {prim_path}")
                analyze_revolute_joint(prim)
            
            if joint_details:
                print(f"\n[Revolute Joint詳細情報] {prim_path}")
                analyze_revolute_joint_details(prim)
                
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_types["PrismaticJoint"].append(prim_path)
            if verbose:
                print(f"\n[Prismatic Joint] {prim_path}")
                analyze_prismatic_joint(prim)
            
            if joint_details:
                print(f"\n[Prismatic Joint詳細情報] {prim_path}")
                analyze_prismatic_joint_details(prim)
                
        elif prim.IsA(UsdPhysics.SphericalJoint):
            joint_types["SphericalJoint"].append(prim_path)
            if verbose:
                print(f"\n[Spherical Joint] {prim_path}")
            
            if joint_details:
                print(f"\n[Spherical Joint詳細情報] {prim_path}")
                analyze_spherical_joint_details(prim)
                
        elif prim.IsA(UsdPhysics.FixedJoint):
            joint_types["FixedJoint"].append(prim_path)
            if verbose:
                print(f"\n[Fixed Joint] {prim_path}")
            
            if joint_details:
                print(f"\n[Fixed Joint詳細情報] {prim_path}")
                analyze_fixed_joint_details(prim)
        
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
                if not verbose and not joint_details:
                    # verbose=FalseかつjointDetails=Falseの場合、最初の3つだけ表示
                    for i, path in enumerate(paths[:3]):
                        print(f"  - {path}")
                    if len(paths) > 3:
                        print(f"  ... 他 {len(paths) - 3} 個")
                elif joint_details:
                    # joint_details=Trueの場合、すべて表示
                    for path in paths:
                        print(f"  - {path}")
        
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

def analyze_physics_joint_details(joint, prim):
    """Physics Jointの詳細情報を表示"""
    print("  === 基本情報 ===")
    print(f"  フルパス: {prim.GetPath()}")
    print(f"  名前: {prim.GetName()}")
    print(f"  タイプ名: {prim.GetTypeName()}")
    
    # Body関係
    body0_rel = joint.GetBody0Rel()
    body1_rel = joint.GetBody1Rel()
    
    print("\n  === ボディ関係 ===")
    if body0_rel.GetTargets():
        print(f"  Body0: {body0_rel.GetTargets()[0]}")
    else:
        print("  Body0: (未設定)")
        
    if body1_rel.GetTargets():
        print(f"  Body1: {body1_rel.GetTargets()[0]}")
    else:
        print("  Body1: (未設定)")
    
    # 位置と方向
    print("\n  === 位置と方向 ===")
    local_pos0 = joint.GetLocalPos0Attr()
    local_pos1 = joint.GetLocalPos1Attr()
    local_rot0 = joint.GetLocalRot0Attr()
    local_rot1 = joint.GetLocalRot1Attr()
    
    if local_pos0.HasValue():
        print(f"  LocalPos0: {local_pos0.Get()}")
    if local_pos1.HasValue():
        print(f"  LocalPos1: {local_pos1.Get()}")
    if local_rot0.HasValue():
        print(f"  LocalRot0: {local_rot0.Get()}")
    if local_rot1.HasValue():
        print(f"  LocalRot1: {local_rot1.Get()}")
    
    # ジョイント有効/無効
    enabled_attr = joint.GetJointEnabledAttr()
    if enabled_attr.HasValue():
        print("\n  === 状態 ===")
        print(f"  有効: {enabled_attr.Get()}")
    
    # すべての属性を表示
    print("\n  === すべての属性 ===")
    for attr in prim.GetAttributes():
        if attr.HasValue():
            print(f"  {attr.GetName()}: {attr.Get()}")

def analyze_revolute_joint_details(prim):
    """Revolute Jointの詳細情報を表示"""
    joint = UsdPhysics.RevoluteJoint(prim)
    
    # 基本のPhysics Joint詳細を表示
    analyze_physics_joint_details(joint, prim)
    
    print("\n  === Revolute Joint固有情報 ===")
    
    # 軸
    axis_attr = joint.GetAxisAttr()
    if axis_attr.HasValue():
        print(f"  回転軸: {axis_attr.Get()}")
    
    # 制限
    lower_limit = joint.GetLowerLimitAttr()
    upper_limit = joint.GetUpperLimitAttr()
    
    if lower_limit.HasValue() and upper_limit.HasValue():
        print(f"  角度制限: [{lower_limit.Get():.2f}, {upper_limit.Get():.2f}] degrees")
    elif lower_limit.HasValue():
        print(f"  下限角度: {lower_limit.Get():.2f} degrees")
    elif upper_limit.HasValue():
        print(f"  上限角度: {upper_limit.Get():.2f} degrees")
    
    # ドライブ関連
    drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
    if drive_api:
        print("\n  === ドライブ設定 ===")
        
        damping = drive_api.GetDampingAttr()
        stiffness = drive_api.GetStiffnessAttr()
        target_position = drive_api.GetTargetPositionAttr()
        target_velocity = drive_api.GetTargetVelocityAttr()
        max_force = drive_api.GetMaxForceAttr()
        
        if damping.HasValue():
            print(f"  ダンピング: {damping.Get()}")
        if stiffness.HasValue():
            print(f"  剛性: {stiffness.Get()}")
        if target_position.HasValue():
            print(f"  目標位置: {target_position.Get()}")
        if target_velocity.HasValue():
            print(f"  目標速度: {target_velocity.Get()}")
        if max_force.HasValue():
            print(f"  最大力: {max_force.Get()}")

def analyze_prismatic_joint_details(prim):
    """Prismatic Jointの詳細情報を表示"""
    joint = UsdPhysics.PrismaticJoint(prim)
    
    # 基本のPhysics Joint詳細を表示
    analyze_physics_joint_details(joint, prim)
    
    print("\n  === Prismatic Joint固有情報 ===")
    
    # 軸
    axis_attr = joint.GetAxisAttr()
    if axis_attr.HasValue():
        print(f"  移動軸: {axis_attr.Get()}")
    
    # 制限
    lower_limit = joint.GetLowerLimitAttr()
    upper_limit = joint.GetUpperLimitAttr()
    
    if lower_limit.HasValue() and upper_limit.HasValue():
        print(f"  移動制限: [{lower_limit.Get():.2f}, {upper_limit.Get():.2f}] units")
    elif lower_limit.HasValue():
        print(f"  下限位置: {lower_limit.Get():.2f} units")
    elif upper_limit.HasValue():
        print(f"  上限位置: {upper_limit.Get():.2f} units")
    
    # ドライブ関連
    drive_api = UsdPhysics.DriveAPI.Apply(prim, "linear")
    if drive_api:
        print("\n  === ドライブ設定 ===")
        
        damping = drive_api.GetDampingAttr()
        stiffness = drive_api.GetStiffnessAttr()
        target_position = drive_api.GetTargetPositionAttr()
        target_velocity = drive_api.GetTargetVelocityAttr()
        max_force = drive_api.GetMaxForceAttr()
        
        if damping.HasValue():
            print(f"  ダンピング: {damping.Get()}")
        if stiffness.HasValue():
            print(f"  剛性: {stiffness.Get()}")
        if target_position.HasValue():
            print(f"  目標位置: {target_position.Get()}")
        if target_velocity.HasValue():
            print(f"  目標速度: {target_velocity.Get()}")
        if max_force.HasValue():
            print(f"  最大力: {max_force.Get()}")

def analyze_spherical_joint_details(prim):
    """Spherical Jointの詳細情報を表示"""
    joint = UsdPhysics.SphericalJoint(prim)
    
    # 基本のPhysics Joint詳細を表示
    analyze_physics_joint_details(joint, prim)
    
    print("\n  === Spherical Joint固有情報 ===")
    
    # 球面ジョイントは通常、軸制限や円錐制限を持つ
    cone_angle_attr = prim.GetAttribute("physics:coneAngle0Limit")
    if cone_angle_attr and cone_angle_attr.HasValue():
        print(f"  円錐角度制限: {cone_angle_attr.Get():.2f} degrees")
    
    # ドライブ関連（もし存在すれば）
    drive_api = UsdPhysics.DriveAPI.Apply(prim, "rotational")
    if drive_api:
        print("\n  === ドライブ設定 ===")
        
        damping = drive_api.GetDampingAttr()
        stiffness = drive_api.GetStiffnessAttr()
        
        if damping.HasValue():
            print(f"  ダンピング: {damping.Get()}")
        if stiffness.HasValue():
            print(f"  剛性: {stiffness.Get()}")

def analyze_fixed_joint_details(prim):
    """Fixed Jointの詳細情報を表示"""
    joint = UsdPhysics.FixedJoint(prim)
    
    # 基本のPhysics Joint詳細を表示
    analyze_physics_joint_details(joint, prim)
    
    print("\n  === Fixed Joint固有情報 ===")
    print("  このジョイントは2つのボディを固定的に結合します。")
    print("  相対的な動きは許可されません。")

def check_alternative_structures(stage):
    """代替のジョイント構造をチェック"""
    
    # すべての関係性をチェック
    relationships = defaultdict(list)
    
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
    parser.add_argument('-j', '--joint-details', action='store_true',
                       help='関節の詳細情報を表示')
    
    args = parser.parse_args()
    
    # ファイルの存在確認
    if not Path(args.usd_file).exists():
        print(f"エラー: ファイルが見つかりません: {args.usd_file}")
        sys.exit(1)
    
    find_all_joints(args.usd_file, args.verbose, args.joint_details)

if __name__ == "__main__":
    main()