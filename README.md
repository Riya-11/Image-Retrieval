### Image-Retrieval

This repository Implements a teacher-student model to perform image retrieval. 

The code is divided into 4 files -
- teacher.py - the training and evaluation script for the teacher model
- distiller.py - consists a Distiller class that distills the knowledge from the teacher model to the student model
- student.py - training script for the student model
- retrieval.py - given a query, it performs retrieval on the cifar-10 dataset. Retrieval results can be evaluated using Average Precision and Mean Average Precision metrics.

![Architecture](https://github.com/Riya-11/Image-Retrieval/blob/main/architecture.png?raw=true)

The above picture shows the training process of teacher model and student model. Teacher model is trained with the cross entropy loss and student model is trained with cross
entropy and knowledge distillation. 

The work done in this repository is an attempt to implement the idea proposed in the following paper:

<a id="1">[1]</a> Zhai, Hongjia, et al. "Deep Transfer Hashing for Image Retrieval." IEEE Transactions on Circuits and Systems for Video Technology (2020).

- [x] Teacher Student Model Implementation
- [ ] Hashing to improve retrieval performance