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
To run project first follow described instructions for
[ctpn](https://github.com/eragonruan/text-detection-ctpn) and
[crnn](https://github.com/MotamedNia/scene-text-recog)
then run
```python
 python detector.py -i <IMAGE_PATH> 
```