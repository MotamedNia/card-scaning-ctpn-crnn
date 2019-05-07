# card-scaning-ctpn-crnn
Implmentation of an card number scaning algorithm using ctpn for detection and crnn for recognition

This project try to read a card image and write card information 
in a csv file

This project based on 
[tensorflow ctpn implementation](https://github.com/eragonruan/text-detection-ctpn) and
[pytorch crnn implementation](https://github.com/MotamedNia/scene-text-recog)


# Requirements
* python 3.5
* tensorflow 
* pytorch
* opencv
* numpy
* pytesseract
* imutils

# Run
checkout the project
```shell
git clone --recurse-submodules https://github.com/MotamedNia/card-scaning-ctpn-crnn.git
```
install tesseract-ocr
```shell
sudo apt-get install tesseract-ocr
```
install requirements. It's better to create a 
virtual environment before install requirements
```shell
pip install -r requirements.txt
```
install pytorch 
```shell
pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
```



follow described instructions for
[ctpn](https://github.com/eragonruan/text-detection-ctpn) and
[crnn](https://github.com/MotamedNia/scene-text-recog)

then run
```python
 python detector.py -i <IMAGE_PATH> 
```