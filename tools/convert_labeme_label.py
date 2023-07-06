from __future__ import print_function

import argparse
import glob
import os
import os.path as osp

import numpy as np
import PIL.Image

import labelme

def lblsave(filename, lbl):
    import imgviz

    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil = lbl_pil.convert("RGB")
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('label_dir', help='label dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    parser.add_argument(
        '--noviz', help='no visualization', action='store_true'
    )
    args = parser.parse_args()

    
    
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1 :
            assert class_name == '__ignore__'
            continue
        elif class_id == 0 :
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    class_name_to_id['other'] = 0

    #create a new dir for converted data
    output_dir = osp.join(os.path.dirname(args.label_dir),'labels_converted')
    os.makedirs(output_dir,exist_ok = True)
    for filename in glob.glob(osp.join(args.label_dir, '*.json')):
        print('Generating dataset from:', filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        
        lblsave(f'{output_dir}/{base}', lbl)
        
if __name__ == '__main__':
    main()