#!/usr/bin/env python3
"""
USD Root Finder Script
Find articulation root in USD files for Isaac Lab/Sim
"""

import argparse
from pathlib import Path
from pxr import Usd, UsdPhysics, UsdGeom, Sdf


def find_articulation_root(usd_path: str) -> None:
    """Find and display articulation root information in USD file."""
    print(f"\n{'='*60}")
    print(f"Analyzing USD file: {usd_path}")
    print(f"{'='*60}\n")
    
    try:
        # Open USD stage
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"Error: Could not open USD file at {usd_path}")
            return
        
        # 1. Check for ArticulationRootAPI
        print("1. Searching for ArticulationRootAPI...")
        articulation_roots = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(prim)
                print(f"   ✓ Found ArticulationRootAPI at: {prim.GetPath()}")
                
                # Get additional info
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(f"     - Also has RigidBodyAPI")
                if prim.GetTypeName():
                    print(f"     - Type: {prim.GetTypeName()}")
        
        if not articulation_roots:
            print("   ✗ No ArticulationRootAPI found")
        
        # 2. Check default prim
        print("\n2. Default Prim Information...")
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            print(f"   ✓ Default prim: {default_prim.GetPath()}")
            print(f"     - Type: {default_prim.GetTypeName()}")
        else:
            print("   ✗ No default prim set")
        
        # 3. Find all rigid bodies (potential roots)
        print("\n3. All Rigid Bodies (potential roots)...")
        rigid_bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(prim)
                # Only show first few to avoid clutter
                if len(rigid_bodies) <= 5:
                    print(f"   - {prim.GetPath()}")
        
        if len(rigid_bodies) > 5:
            print(f"   ... and {len(rigid_bodies) - 5} more rigid bodies")
        
        # 4. Find joints and their relationships
        print("\n4. Joint Information...")
        joints = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joints.append(prim)
                if len(joints) <= 3:  # Show first few joints
                    joint = UsdPhysics.Joint(prim)
                    body0_targets = joint.GetBody0Rel().GetTargets()
                    body1_targets = joint.GetBody1Rel().GetTargets()
                    print(f"   - Joint: {prim.GetPath()}")
                    if body0_targets:
                        print(f"     Body0: {body0_targets[0]}")
                    if body1_targets:
                        print(f"     Body1: {body1_targets[0]}")
        
        if len(joints) > 3:
            print(f"   ... and {len(joints) - 3} more joints")
        
        # 5. Hierarchy analysis
        print("\n5. Hierarchy Analysis...")
        # Find bodies that are not body1 of any joint (potential roots)
        all_body1_paths = set()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                body1_targets = joint.GetBody1Rel().GetTargets()
                for target in body1_targets:
                    all_body1_paths.add(str(target))
        
        potential_roots = []
        for body in rigid_bodies:
            if str(body.GetPath()) not in all_body1_paths:
                potential_roots.append(body)
        
        print(f"   Bodies that are not child of any joint (potential roots):")
        for root in potential_roots[:3]:  # Show first 3
            print(f"   - {root.GetPath()}")
        
        # 6. Summary
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        
        if articulation_roots:
            print(f"✓ Articulation Root: {articulation_roots[0].GetPath()}")
        elif potential_roots:
            print(f"✓ Most likely root (no parent joint): {potential_roots[0].GetPath()}")
        elif default_prim and default_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"✓ Default prim with RigidBody: {default_prim.GetPath()}")
        elif rigid_bodies:
            print(f"✓ First rigid body found: {rigid_bodies[0].GetPath()}")
        else:
            print("✗ No clear root found")
        
        print(f"\nTotal counts:")
        print(f"  - Rigid bodies: {len(rigid_bodies)}")
        print(f"  - Joints: {len(joints)}")
        print(f"  - Articulation roots: {len(articulation_roots)}")
        
    except Exception as e:
        print(f"Error analyzing USD file: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Find articulation root in USD files for Isaac Lab/Sim"
    )
    parser.add_argument(
        "usd_path",
        type=str,
        help="Path to the USD file to analyze"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information about all prims"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.usd_path).exists():
        print(f"Error: File not found: {args.usd_path}")
        return
    
    # Find root
    find_articulation_root(args.usd_path)


if __name__ == "__main__":
    main()