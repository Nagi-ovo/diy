import re
import sys
import os

def replace_image_links(content):
    """
    替换Markdown文件中所有符合![[路径|大小]]格式的图片链接为![][路径]，
    并将路径中的空格替换为%20。
    
    :param content: 原始Markdown文件内容
    :return: 修改后的Markdown文件内容
    """
    # 正则表达式模式：
    # ![[路径|大小]]
    pattern = r'!\[\[(.*?)\|(.*?)\]\]'
    
    def replacement(match):
        path = match.group(1)
        path_encoded = path.replace(' ', '%20')
        return f'![]({path_encoded})'
    
    # 使用re.sub进行替换
    new_content = re.sub(pattern, replacement, content)
    return new_content

def process_file(input_file, output_file):
    """
    读取输入Markdown文件，处理内容，并将结果写入输出文件。
    
    :param input_file: 输入Markdown文件路径
    :param output_file: 输出Markdown文件路径
    """
    if not os.path.isfile(input_file):
        print(f"错误：文件 '{input_file}' 不存在。")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = replace_image_links(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"处理完成！已将修改后的内容保存到 '{output_file}'。")

def main():
    """
    主函数，处理命令行参数并执行文件转换。
    """
    if len(sys.argv) != 3:
        print("使用方法：python imgre.py 输入文件.md 输出文件.md")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_file(input_file, output_file)

if __name__ == "__main__":
    main()
