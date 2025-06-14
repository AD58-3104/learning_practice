from itertools import combinations

def generate_combinations_as_lists(item_list):
    """
    組み合わせをリストとして返すバージョン
    """
    combinations_list = []
    
    # 1個組の組み合わせ
    for item in item_list:
        combinations_list.append([item])
    
    # 2個組の組み合わせ
    for combo in combinations(item_list, 2):
        combinations_list.append(list(combo))
    
    return combinations_list


def generate_joint_strings(string_list, float_val:float):
    """
    文字列のリストと浮動小数点のリストから、
    0を浮動小数点の値に、"right_hip_roll"を文字列の値に置き換えた文字列を生成する
    
    Args:
        string_list: 文字列のリスト
        float_list: 浮動小数点のリスト
    
    Returns:
        生成された文字列のリスト
    """
    result_strings = []
    
    # 各組み合わせに対して文字列を生成
    # for string_val in string_list:
    #     for float_val in float_list:
            # テンプレート文字列で0とright_hip_rollを置き換え
    template = """  - joint_torque: [0]
    joint_names: ["right_hip_roll"]"""
        
        # 置き換えを実行
    new_string = template.replace("0", ','.join(map(str,[float_val] * len(string_list)))).replace('"right_hip_roll"', ','.join(map(lambda s: f'"{s}"',string_list)))
    result_strings.append(new_string)
    
    return result_strings


if __name__ == "__main__":
    # サンプルデータ
    string_elem_list = [
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_joint"
    ]
    float_elem_list = [0,50,100,150,200,250,300]
    
    # string_lists = generate_combinations_as_lists(string_elem_list)

    iterate_num = 0
    for num in float_elem_list:
        for names in string_elem_list:
            # 文字列生成
            generated_strings = generate_joint_strings([names], num)
            
            # 結果を表示
            for i, generated_string in enumerate(generated_strings, 1):
                print(generated_string)
                iterate_num += 1
    print(f"Iterate {iterate_num} times...")