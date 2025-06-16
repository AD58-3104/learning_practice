#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

print("スクリプト開始...")

# インポートの確認
try:
    from pxr import Usd, UsdSkel, UsdGeom, Tf
    print("✓ pxr モジュールのインポート成功")
    print(f"  USD バージョン: {Usd.GetVersion()}")
except ImportError as e:
    print(f"✗ pxr モジュールのインポート失敗: {e}")
    sys.exit(1)

def debug_usd_file(usd_file_path):
    """USDファイルの詳細なデバッグ情報を表示"""
    
    print(f"\n--- ファイル情報 ---")
    print(f"パス: {usd_file_path}")
    print(f"存在: {Path(usd_file_path).exists()}")
    print(f"サイズ: {Path(usd_file_path).stat().st_size} bytes")
    
    # ステージを開く
    print(f"\n--- ステージを開いています ---")
    try:
        stage = Usd.Stage.Open(str(usd_file_path))
        if stage:
            print("✓ ステージのオープン成功")
        else:
            print("✗ ステージのオープン失敗")
            return
    except Exception as e:
        print(f"✗ エラー: {e}")
        return
    
    # ステージ情報
    print(f"\n--- ステージ情報 ---")
    print(f"ルートプリム: {stage.GetPseudoRoot()}")
    print(f"デフォルトプリム: {stage.GetDefaultPrim()}")
    print(f"タイムコード範囲: {stage.GetTimeCodesPerSecond()}")
    
    # すべてのプリムを列挙
    print(f"\n--- プリム一覧 ---")
    prim_count = 0
    skeleton_count = 0
    mesh_count = 0
    xform_count = 0
    
    for prim in stage.Traverse():
        prim_count += 1
        prim_type = prim.GetTypeName()
        
        # 最初の10個のプリムを表示
        if prim_count <= 10:
            print(f"  [{prim_count}] {prim.GetPath()} (Type: {prim_type})")
        
        # 型別にカウント
        if prim.IsA(UsdSkel.Skeleton):
            skeleton_count += 1
            print(f"\n🦴 スケルトン発見: {prim.GetPath()}")
            analyze_skeleton(prim)
        elif prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
        elif prim.IsA(UsdGeom.Xform):
            xform_count += 1
    
    if prim_count > 10:
        print(f"  ... 他 {prim_count - 10} 個のプリム")
    
    print(f"\n--- 統計 ---")
    print(f"総プリム数: {prim_count}")
    print(f"スケルトン: {skeleton_count}")
    print(f"メッシュ: {mesh_count}")
    print(f"Xform: {xform_count}")
    
    # スケルトンが見つからない場合の追加チェック
    if skeleton_count == 0:
        print(f"\n--- スケルトンが見つからない場合の追加チェック ---")
        check_for_skeleton_api(stage)

def analyze_skeleton(prim):
    """スケルトンの詳細を分析"""
    skeleton = UsdSkel.Skeleton(prim)
    
    # 各属性をチェック
    print(f"  属性一覧:")
    for attr in prim.GetAttributes():
        print(f"    - {attr.GetName()}: {attr.GetTypeName()}")
    
    # ジョイント情報
    joints_attr = skeleton.GetJointsAttr()
    if joints_attr and joints_attr.HasValue():
        joints = joints_attr.Get()
        print(f"  ジョイント数: {len(joints) if joints else 0}")
        if joints and len(joints) > 0:
            print(f"  最初のジョイント: {joints[0]}")
    else:
        print(f"  ジョイント属性が設定されていません")

def check_for_skeleton_api(stage):
    """SkelAPIを使用しているプリムをチェック"""
    api_count = 0
    
    for prim in stage.Traverse():
        # SkelBindingAPIをチェック
        if prim.HasAPI(UsdSkel.BindingAPI):
            api_count += 1
            print(f"  SkelBindingAPI: {prim.GetPath()}")
            
            # スケルトン関係を取得
            binding = UsdSkel.BindingAPI(prim)
            skel_rel = binding.GetSkeletonRel()
            if skel_rel and skel_rel.GetTargets():
                print(f"    → スケルトン参照: {skel_rel.GetTargets()}")
        
        # アニメーションをチェック
        if prim.IsA(UsdSkel.Animation):
            print(f"  SkelAnimation: {prim.GetPath()}")
    
    if api_count == 0:
        print("  SkelAPIを使用しているプリムは見つかりませんでした")

def main():
    parser = argparse.ArgumentParser(description='USDファイルのデバッグ情報を表示')
    parser.add_argument('usd_file', help='チェックするUSDファイル')
    args = parser.parse_args()
    
    debug_usd_file(args.usd_file)

if __name__ == "__main__":
    main()