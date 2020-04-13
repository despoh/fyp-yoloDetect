//
//  PredictionLayer.swift
//  fyp-yoloDetect
//
//  Created by Kee Heng on 31/03/2020.
//  Copyright © 2020 Kee Heng. All rights reserved.
//

import Foundation
import UIKit

class BoundingBoxLayer{

    let labels = ["Glass", "Metal", "Paper", "Plastic"]

    var previewFrame : CGRect?

    let colors: [CGColor] = [UIColor.systemRed.cgColor,UIColor.systemBlue.cgColor,UIColor.systemGreen.cgColor,UIColor.systemPurple.cgColor]


    var layer: CAShapeLayer

     init() {
       layer = CAShapeLayer()
       layer.fillColor = UIColor.clear.cgColor
     }

     func addBoundingBoxes(predictions: [Yolov3.Prediction]) {
       layer.sublayers = nil
       for prediction in predictions {
        let boundingBox = BoundingBox(predRect: prediction.rect,previewFrame: previewFrame!, label: labels[prediction.classIndex], confidence: prediction.score, color: colors[prediction.classIndex])

        boundingBox.addTo(layer: layer)
       }

     }

}

struct BoundingBox {

    let layer = CAShapeLayer()
    let textLayer = CATextLayer()

    init (predRect: CGRect, previewFrame: CGRect,label: String, confidence: Float,color: CGColor) {

     let width = previewFrame.width
     let height = width * 16/9

     let ratioX = width / CGFloat(Yolov3.inputSize)
     let ratioY = height / CGFloat(Yolov3.inputSize)

      let rect = CGRect(x: predRect.origin.x * ratioX, y: predRect.origin.y * ratioY , width: predRect.width * ratioX ,height: predRect.height * ratioY)

      layer.fillColor = color
      layer.opacity = 0.5
      layer.lineWidth = 2

      layer.bounds =  CGRect(x: 0, y:0, width: rect.width  , height: rect.height )
      layer.position = CGPoint(x: rect.midX, y:  rect.midY)
      layer.backgroundColor = color
      layer.cornerRadius = 10

      textLayer.foregroundColor = UIColor.black.cgColor
      textLayer.fontSize = 13
      textLayer.font = UIFont(name: "Helvetica", size: textLayer.fontSize)
      textLayer.alignmentMode = CATextLayerAlignmentMode.left
      textLayer.frame = CGRect(x: rect.origin.x+10, y: rect.origin.y+10,
                               width: 85, height: 18)
      textLayer.backgroundColor = UIColor.clear.cgColor
      textLayer.string = "\(label):" + String(format: "%.2f", confidence)
    }

    func addTo(layer: CALayer) {
      layer.addSublayer(self.layer)
      layer.addSublayer(self.textLayer)
    }
  }
