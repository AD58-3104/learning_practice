#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import os

# 利用可能なバックエンドを試す
def setup_matplotlib():
    import matplotlib
    
    # GUIバックエンドのリスト（優先順位順）
    gui_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTKAgg', 'WXAgg']
    
    for backend in gui_backends:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            # テスト用のfigureを作成してみる
            fig = plt.figure()
            plt.close(fig)
            print(f"使用するバックエンド: {backend}")
            return plt, backend
        except Exception as e:
            continue
    
    # すべて失敗した場合
    print("警告: インタラクティブなバックエンドが見つかりません。")
    print("以下のコマンドで必要なパッケージをインストールしてください：")
    print("  Ubuntu/Debian: sudo apt-get install python3-tk")
    print("  Fedora: sudo dnf install python3-tkinter")
    print("  pip: pip install PyQt5")
    
    # フォールバック
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt, 'Agg'

# matplotlibのセットアップ
plt, backend_used = setup_matplotlib()

def plot_sech_function(l_value):
    """
    2/(e^(lx) + e^(-lx)) のグラフを描画する関数
    
    Parameters:
    l_value (float): 関数のパラメータl
    """
    # xの範囲を設定
    x = np.linspace(-1, 5, 1000)
    
    # 関数の計算
    y = 2 / (np.exp(l_value * (x)) + np.exp(-l_value * (x))) 
    
    # グラフの作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line, = ax.plot(x, y, 'b-', linewidth=2, label=f'f(x) = 2/(e^({l_value}x) + e^(-{l_value}x))')
    
    # グラフの装飾
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f' [l = {l_value}]', fontsize=14)
    ax.legend(fontsize=10)
    
    # x軸とy軸を表示
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # y軸の範囲を調整
    ax.set_ylim(-0.1, 2.1)

    
    plt.tight_layout()
    
    # 表示またはファイル保存
    if backend_used != 'Agg':
        try:
            plt.show()
            return None
        except Exception as e:
            print(f"表示エラー: {e}")
    
    # Aggバックエンドの場合、またはshow()が失敗した場合
    output_file = f'sech_function_l_{l_value:.2f}.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    abs_path = os.path.abspath(output_file)
    print(f"\nグラフを保存しました: {abs_path}")
    
    return output_file

def install_gui_backend():
    """GUIバックエンドのインストール方法を表示"""
    print("\nインタラクティブなグラフ表示のセットアップ方法：")
    print("\n1. Tkinter (最も簡単):")
    print("   Ubuntu/Debian: sudo apt-get install python3-tk")
    print("   Fedora: sudo dnf install python3-tkinter")
    print("   Arch: sudo pacman -S tk")
    print("\n2. PyQt5 (高機能):")
    print("   pip install PyQt5")
    print("\n3. 環境変数の確認:")
    print("   echo $DISPLAY  # :0 などが表示されるべき")
    print("\n4. SSH経由の場合:")
    print("   ssh -X username@hostname  # X11転送を有効に")

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='2/(e^(lx) + e^(-lx))のグラフを描画します')
    parser.add_argument('l', type=float, help='パラメータlの値')
    parser.add_argument('--install-info', action='store_true', 
                       help='GUIバックエンドのインストール情報を表示')
    
    args = parser.parse_args()
    
    if args.install_info:
        install_gui_backend()
        return
    
    # グラフの描画
    output_file = plot_sech_function(args.l)
    
    # 関数の性質を表示
    print(f"\n関数の性質 (l = {args.l}):")
    print(f"- 最大値: x = 0 での f(0) = {2 / (np.exp(0) + np.exp(0)):.4f}")
    print(f"- 関数は偶関数です（f(-x) = f(x)）")
    if args.l > 0:
        print(f"- x → ±∞ のとき、f(x) → 0")
    elif args.l < 0:
        print(f"- l < 0 の場合も同様の形状ですが、|l|で決まります")
    
    if backend_used == 'Agg':
        print("\nヒント: インタラクティブな表示を有効にするには:")
        print("  python kernel_plot.py --install-info")

if __name__ == "__main__":
    main()
