from pathlib import Path

SUPPORTED_EXTENSIONS={".txt",".md"}

def load_document(file_path:str)->dict:
    """Load a text or Markdown document and return its content and metadata."""
    path = Path(file_path)      #path是由Path类定义的对象

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:             #先拿path.suffix得到字符串，在对这个字符串调用.lower方法
        raise ValueError(f"Unsupported file type: {path.suffix}")

    content = path.read_text(encoding="utf-8")

    return {
        "source": path.name,
        "content": content,
        "char_count": len(content),
    }

"""**主程序入口**：
当这个脚本文件直接运行时，下面代码才执行；如果这个文件被别的脚本`import`导入，这一块代码不会执行。
作用：分离功能代码和测试代码，写模块时非常常用。"""
if __name__=="__main__":
    project_root=Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "sample.md"
    document = load_document(str(sample_path))      #document表示函数load_document的返回值

    print(f"Source: {document['source']}")
    print(f"Characters: {document['char_count']}")
    print("Content:")
    print(document["content"])


