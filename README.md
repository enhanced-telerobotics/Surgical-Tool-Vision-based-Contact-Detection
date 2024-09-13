# Telesurgical Tool Vision-based Contact Detection

![Pipeline](flowchart.png)

This project is part of the research work titled **“Vision-Based Force Estimation for Minimally Invasive Telesurgery Through Contact Detection and Local Stiffness Models”**. The project utilizes an open-source silicone dataset of simulated palpation using surgical robot end effectors. Two neural network binary classifiers were trained based on contact states obtained through sensor thresholds or crowdsourcing (MTurk).

## Getting Started

### Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Run the Demo

1. **Download the Silicone Dataset**: [Download Link](https://github.com/enhanced-telerobotics/single_psm_manipulation_dataset)

   Make sure to unpack and rename the datasets using each bag name. Your file structure should look like the following:

   ```
   Path_to_root
   ├── R1_M1_T1_1
   │   ├── labels_30hz.txt
   │   ├── img_0.jpg
   │   ├── img_1.jpg
   │   └── ...
   ```

2. **Download Pre-generate DeepLabCut Keypoints Tracking Sheets** for the silicone dataset: [Download Link](https://drive.google.com/file/d/1kkjRnffpC0ZWf0m7RvENEb1KkhbDvi4v/view?usp=sharing)

3. **Run the Demo Notebook**: Open and run [`silicone-demo.ipynb`](silicone-demo.ipynb) for the model training and evaluation pipeline. Ensure you modify [`settings/paths.json`](settings/paths.json) to reflect your local paths correctly.

## Citation

If you find this project or the associated paper helpful in your research, please cite it as follows:

```
@article{Yang_2024, 
  title={Vision-Based Force Estimation for Minimally Invasive Telesurgery Through Contact Detection and Local Stiffness Models}, 
  ISSN={2424-9068}, 
  url={http://dx.doi.org/10.1142/S2424905X24400087}, 
  DOI={10.1142/s2424905x24400087}, 
  journal={Journal of Medical Robotics Research}, 
  publisher={World Scientific Pub Co Pte Ltd}, 
  author={Yang, Shuyuan and Le, My H. and Golobish, Kyle R. and Beaver, Juan C. and Chua, Zonghe}, 
  year={2024}, 
  month={Jul}
}
```

## Contact

For any questions, please feel free to email [sxy841@case.edu](mailto:sxy841@case.edu).
