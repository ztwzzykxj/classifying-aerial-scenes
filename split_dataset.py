import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.8):
    random.seed(42)

    categories = os.listdir(source_dir)
    for category in categories:
        category_path = os.path.join(source_dir, category)
        if not os.path.isdir(category_path):
            continue

        images = os.listdir(category_path)
        random.shuffle(images)

        split_point = int(len(images) * train_ratio)
        train_images = images[:split_point]
        test_images = images[split_point:]

        # 创建 train 和 test 子目录
        for split, split_images in [('train', train_images), ('test', test_images)]:
            split_folder = os.path.join(dest_dir, split, category)
            os.makedirs(split_folder, exist_ok=True)

            for img in split_images:
                src = os.path.join(category_path, img)
                dst = os.path.join(split_folder, img)
                shutil.copy2(src, dst)

    print(f"数据划分完成！train 和 test 路径已写入 {dest_dir}")

# ===== 用法 =====
source_folder = r"D:\archive\Aerial_Landscapes"
output_folder = r"D:\archive\Aerial_Landscapes_Split"

split_dataset(source_folder, output_folder, train_ratio=0.8)
