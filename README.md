# Transparency_sam

To support our experiments, we constructed a custom synthetic dataset that reflects real-world scenarios by varying environments and viewing angles.  
This dataset was built using the **TransPose** dataset, which provides images and 3D models of transparent equipment and objects.  
These 3D models were imported into **Unreal Engine** to generate a total of **4,350** paired RGB images and corresponding ground truth segmentation maps.

The dataset is split into:

- **Train**: 3,045 images  
- **Validation**: 652 images  
- **Test**: 653 images

📥 **Download the dataset**:  
[Google Drive Link](https://drive.google.com/file/d/1k0yjpnDzNKcBmW-z1lpChnM80eWi2Mkt/view?usp=sharing)

---

## 📄 License

This dataset is shared **for non-commercial research use only**, in accordance with the license of the original TransPose dataset.  
It is licensed under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike 4.0 International).

For more information, please refer to the original TransPose dataset and its license terms.

---

## 📚 Citation

If you use this dataset, please cite the original TransPose paper:

```bibtex
@article{doi:10.1177/02783649231213117,
  author  = {Jeongyun Kim and Myung-Hwan Jeon and Sangwoo Jung and Wooseong Yang and Minwoo Jung and Jaeho Shin and Ayoung Kim},
  title   = {TRansPose: Large-scale multispectral dataset for transparent object},
  journal = {The International Journal of Robotics Research},
  volume  = {43},
  number  = {6},
  pages   = {731--738},
  year    = {2024},
  doi     = {10.1177/02783649231213117},
  url     = {https://doi.org/10.1177/02783649231213117}
}

}
