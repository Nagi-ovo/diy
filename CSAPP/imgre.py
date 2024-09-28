import re
import os

def replace_image_links(content):
    """
    替换Markdown文件中所有符合![[路径|大小]]格式的图片链接为![](路径)，
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

    new_content = re.sub(pattern, replacement, content)
    return new_content

def backup_file(file_path):
    """
    创建文件的备份副本。
    
    :param file_path: 原始文件路径
    """
    backup_path = f"{file_path}.bak"
    try:
        with open(file_path, 'r', encoding='utf-8') as original_file:
            content = original_file.read()
        with open(backup_path, 'w', encoding='utf-8') as backup_file:
            backup_file.write(content)
        print(f"备份创建成功：{backup_path}")
    except Exception as e:
        print(f"备份失败：{file_path}. 错误：{e}")

def delete_backup(file_path):
    """
    删除文件的备份副本。
    
    :param file_path: 原始文件路径
    """
    backup_path = f"{file_path}.bak"
    try:
        if os.path.isfile(backup_path):
            os.remove(backup_path)
            print(f"备份文件已删除：{backup_path}")
    except Exception as e:
        print(f"删除备份文件失败：{backup_path}. 错误：{e}")

def process_markdown_file(file_path, delete_bak=False):
    """
    处理单个Markdown文件，替换图片链接并保存修改。
    
    :param file_path: Markdown文件路径
    :param delete_bak: 是否删除备份文件
    """
    print(f"正在处理文件：{file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = replace_image_links(content)

        if new_content != content:
            backup_file(file_path)  # 创建备份
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"文件已更新：{file_path}")
            if delete_bak:
                delete_backup(file_path)  # 删除备份
        else:
            print(f"没有需要更新的内容：{file_path}")
    except Exception as e:
        print(f"处理文件时出错：{file_path}. 错误：{e}")

def process_all_markdown_files(directory, delete_bak=False):
    """
    遍历指定目录中的所有Markdown文件并进行处理。
    
    :param directory: 要遍历的目录路径
    :param delete_bak: 是否删除备份文件
    """
    print(f"开始处理目录中的所有Markdown文件：{directory}\n")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.md'):
                file_path = os.path.join(root, file)
                process_markdown_file(file_path, delete_bak)
    print("\n所有文件处理完成。")

def main():
    """
    主函数，执行脚本逻辑。
    """
    # 获取当前目录
    current_directory = os.getcwd()
    
    # 是否删除备份文件
    # 设置为True将删除所有创建的备份文件
    # 设置为False将保留备份文件
    delete_backups = True  # 修改为False以保留备份文件
    
    process_all_markdown_files(current_directory, delete_backups)

if __name__ == "__main__":
    main()
