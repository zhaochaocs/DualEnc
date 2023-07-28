# DualEnc

This is the codebase for the ACL-20 paper 
[Bridging the Structural Gap Between Encoding and Decoding for Data-To-Text Generation](https://www.aclweb.org/anthology/2020.acl-main.224/)

Some codes are borrowed from [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) and [graph-to-text](https://github.com/diegma/graph-2-text)

### Reproduce

To reproduce the results, Please download data from [here](https://github.com/ThiagoCF05/webnlg/tree/master/data/v1.5/en) and then put the three folders under `data/webnlg`.

To train the neural planner, run

`sh planning.sh`

To train & test the PlanEnc model, run

`sh pipeline_PlanEnc.sh`

To train & test the DualEnc model, run

`sh pipeline_DualEnc.sh`


### Citation
```
@inproceedings{zhao2020bridging,
  title={Bridging the structural gap between encoding and decoding for data-to-text generation},
  author={Zhao, Chao and Walker, Marilyn and Chaturvedi, Snigdha},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={2481--2491},
  year={2020}
}
```


