#!/usr/bin/env python
"""
USDファイルから物理ボディ（body）の名前一覧を取得するスクリプト
"""

from pxr import Usd, UsdPhysics, UsdGeom
import sys
import argparse


def get_physics_bodies(stage):
    """
    USDステージから物理ボディの一覧を取得
    
    Args:
        stage: Usd.Stage object
        
    Returns:
        list: 物理ボディのパスと名前のタプルのリスト
    """
    bodies = []
    
    # ステージ内の全プリムを走査
    for prim in stage.Traverse():
        # UsdPhysics.RigidBodyAPI が適用されているかチェック
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_api = UsdPhysics.RigidBodyAPI(prim)
            bodies.append({
                'path': str(prim.GetPath()),
                'name': prim.GetName(),
                'type': 'RigidBody'
            })
        
        # UsdPhysics.ArticulationRootAPI が適用されているかチェック（関節接続されたボディ）
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            bodies.append({
                'path': str(prim.GetPath()),
                'name': prim.GetName(),
                'type': 'ArticulationRoot'
            })
        
        # UsdPhysics.CollisionAPI が適用されているかチェック（コリジョンボディ）
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            # RigidBodyAPIと重複しない場合のみ追加
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append({
                    'path': str(prim.GetPath()),
                    'name': prim.GetName(),
                    'type': 'CollisionOnly'
                })
    
    return bodies


def get_mesh_bodies(stage):
    """
    物理属性を持たないメッシュボディも含めて取得する場合
    
    Args:
        stage: Usd.Stage object
        
    Returns:
        list: メッシュボディのパスと名前のタプルのリスト
    """
    meshes = []
    
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append({
                'path': str(prim.GetPath()),
                'name': prim.GetName(),
                'type': 'Mesh'
            })
    
    return meshes


def main():
    parser = argparse.ArgumentParser(description='USDファイルからbodyの名前一覧を取得')
    parser.add_argument('usd_file', help='USDファイルのパス')
    parser.add_argument('--include-meshes', action='store_true', 
                       help='物理属性を持たないメッシュも含める')
    parser.add_argument('--format', choices=['simple', 'detailed'], default='simple',
                       help='出力フォーマット (default: simple)')
    
    args = parser.parse_args()
    
    # USDファイルを開く
    try:
        stage = Usd.Stage.Open(args.usd_file)
    except Exception as e:
        print(f"エラー: USDファイルを開けません - {e}")
        sys.exit(1)
    
    if not stage:
        print("エラー: ステージの作成に失敗しました")
        sys.exit(1)
    
    # 物理ボディを取得
    physics_bodies = get_physics_bodies(stage)
    
    # メッシュも含める場合
    all_bodies = physics_bodies.copy()
    if args.include_meshes:
        mesh_bodies = get_mesh_bodies(stage)
        # 物理ボディと重複しないメッシュのみ追加
        physics_paths = {body['path'] for body in physics_bodies}
        for mesh in mesh_bodies:
            if mesh['path'] not in physics_paths:
                all_bodies.append(mesh)
    
    # 結果を表示
    if not all_bodies:
        print("物理ボディが見つかりませんでした")
        return
    
    print(f"\n{'='*50}")
    print(f"ファイル: {args.usd_file}")
    print(f"見つかったボディ数: {len(all_bodies)}")
    print(f"{'='*50}\n")
    
    if args.format == 'simple':
        # シンプルな名前のリスト表示
        for body in sorted(all_bodies, key=lambda x: x['name']):
            print(body['name'])
    else:
        # 詳細表示
        for body in sorted(all_bodies, key=lambda x: x['path']):
            print(f"名前: {body['name']}")
            print(f"パス: {body['path']}")
            print(f"タイプ: {body['type']}")
            print("-" * 30)


def get_body_names_from_usd(usd_file_path, include_meshes=False):
    """
    他のスクリプトから呼び出し可能な関数
    
    Args:
        usd_file_path (str): USDファイルのパス
        include_meshes (bool): メッシュも含めるかどうか
        
    Returns:
        list: ボディ名のリスト
    """
    stage = Usd.Stage.Open(usd_file_path)
    if not stage:
        return []
    
    bodies = get_physics_bodies(stage)
    
    if include_meshes:
        mesh_bodies = get_mesh_bodies(stage)
        physics_paths = {body['path'] for body in bodies}
        for mesh in mesh_bodies:
            if mesh['path'] not in physics_paths:
                bodies.append(mesh)
    
    return [body['name'] for body in bodies]


if __name__ == "__main__":
    main()