from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest1.mp4')
parser.add_argument('--algo', type=str, help='Background subtraction method (FD[Frame Differencing], MD[Median Filter], MOG2, KNN.', default='FD')
parser.add_argument('--th', type=int, help='Treshold for background difference.', default=16)
args = parser.parse_args()


th = args.th
## [create]
#create Background Subtractor objects
if args.algo == 'FD':
    '''
    Método não recursivo.
    O background estimado é apenas o frame anterior.
    Apenas tiramos a diferenca entre o frame atual e o anterior.
    '''
    class BackgroundSubtractorFD():
        def apply(self, frame):
            fgMask = cv.absdiff(frame, old_frame)
            _, fgMask = cv.threshold(fgMask, th, 255,cv.THRESH_BINARY)
            return fgMask
    backSub = BackgroundSubtractorFD()
elif args.algo == 'MD':
    '''
    Método não recursivo.
    O background estimado é apenas a mediada de cada pixel de todo o frame.
    Assumimos que os pixels do background permanecem no mesmo lugar em pelo menos metade dos frames.
    '''
    class BackgroundSubtractorMD():
        def apply(self, frame):
            kernel = np.ones((5,5))/25.
            old_frame = cv.filter2D(frame, -1, kernel)
            fgMask = cv.absdiff(frame, old_frame)
            _, fgMask = cv.threshold(fgMask, th, 255,cv.THRESH_BINARY)
            return fgMask
    backSub = BackgroundSubtractorMD()
elif args.algo == 'MOG2':
    '''
    Este método modela cada pixel do background a partir de uma mistura de distribuições gaussianas.
    O algoritimo seleciona a quantidade apropriada de distribuições para cada pixel.
    O peso da mistura de distribuições representa a proporção de tempo em que as cores permanecem na cena.
    As cores do provável background são as que permanecem estaticas por mais frames.
    '''
    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=th, detectShadows=False)
else:
    '''
    Também uma melhoria no método MOG(Mixture of Gaussians).
    Utiliza o método KNN para melhor estimar a densidade do kernel e decidir a quantidade de distribuições por pixel.
    '''
    backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)

## [create]

## [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

_, old_frame = capture.read()
old_frame = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

while True:

    ret, frame = capture.read()

    if frame is None:
        break

    ## [apply]
    #update the background model

    fgMask = backSub.apply(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    fgMask = np.stack((fgMask,)*3, axis=-1)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
        cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    img = cv.hconcat([frame,fgMask])
    img = cv.resize(img,(int(1800),int(900)))
    cv.imshow('Frame', img)
    #cv.imshow('FG Mask', fgMask)
    ## [show]
    old_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
   
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
