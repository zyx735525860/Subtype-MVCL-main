from PIL import Image


def concat_images_grid(image_paths, row_size=5):
    # 将图片分成多行，每行 row_size 张
    rows = [image_paths[i:i + row_size] for i in range(0, len(image_paths), row_size)]

    # 水平拼接每一行的图片
    row_images = []
    for row in rows:
        images = [Image.open(path) for path in row]
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        row_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width

        row_images.append(row_image)

    # 垂直拼接所有行图片
    widths, heights = zip(*(img.size for img in row_images))
    max_width = max(widths)
    total_height = sum(heights)

    concatenated_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for row_img in row_images:
        concatenated_image.paste(row_img, (0, y_offset))
        y_offset += row_img.height

    return concatenated_image

image_paths = ["C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\BRCA.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\BLCA.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\GBM.png" ,
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\KIRC.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\LUAD.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\PAAD.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\SKCM.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\STAD.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\UCEC.png",
               "C:\\Users\\one\\Desktop\\linshi\\model\\denoise_omics\\T_SNE\\UVM.png"
              ]
concatenated_image = concat_images_grid(image_paths)
concatenated_image.show()
