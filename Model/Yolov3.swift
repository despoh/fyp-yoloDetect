//
//  Yolov3.swift
//  fyp-yoloDetect
//
//  Created by Kee Heng on 31/03/2020.
//  Copyright © 2020 Kee Heng. All rights reserved.
//

import Foundation
import CoreML
import Accelerate
import UIKit
import Vision

class Yolov3:NSObject{

    let anchors :[String: Array<Float>]! = [
      "output1": [228,184, 285,359, 341,260],
      "output2": [136,129, 142,363, 203,290],
      "output3": [55,69, 75,234, 133,240]
    ]


    static let inputSize : Float = 512.0

    var predictions : [Prediction]?
    var requests = [VNCoreMLRequest]()

    struct Prediction {
        let classIndex: Int
        let score: Float
        let rect: CGRect
      }

    override init(){
        super.init()
        if let url = Bundle.main.url(forResource:"yolo", withExtension: "mlmodelc"){
                do{
                        let yoloModel = try MLModel(contentsOf: url)
                        setUPVision(mlmodel:yoloModel)
                }catch let error{
                       print("Error: ",error);
                }

               }else{
                   print("model not found");
               }

    }
    
    //integrate coreml model with vision, completehandler "processRequest"
    //is used when function predict is called
    
    func setUPVision(mlmodel:MLModel){
        let visionModel = try? VNCoreMLModel(for: mlmodel)
        let request = VNCoreMLRequest(model: visionModel!, completionHandler: processRequest)

        request.imageCropAndScaleOption = .scaleFill

        requests.append(request)
    }
    

    //vision framework preprocess the image in pixelBuffer
    // and make inference
    // result is returned as the parameter in completeHandler stated in setUPVision

    
    func predict(pixelBuffer: CVPixelBuffer){

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
           DispatchQueue.global().async {
            try? handler.perform(self.requests)
              }
    }

    //get prediction results and process them
    //append the processed result to array which is used to render Bounding Boxes 
    //NMS is used to filter out prediciton with overlapped bounding boxes with the same class
    
    func processRequest(request: VNRequest, error: Error?){
        if let observations = request.results as? [VNCoreMLFeatureValueObservation]{
            predictions = [Prediction]()
            try? observations.forEach { (observation) in
                let feature = observation.featureValue.multiArrayValue
                let prediction = try extractFeatures(output: feature!,name:observation.featureName)
                    predictions?.append(contentsOf: prediction)
                    nonMaxSuppression(boxes: &(predictions!), threshold: 0.4)

            }
        }
        

        
    }
    
    //512x 512 image divided into 13x13, 26x26, 52x52 grids.
    // loop through each grid and for each grid loop through 3 bounding boxes
    // each box contain 9 features [x,y,w,h,confidence of objectness, probability of glass ,probability of metal, probabilty of paper, probabilty of plastic]
    // so each grid will contain 9x3 = 27 features
    // so in a 13x13 grids, there is a (13 x 13 x 27) features in the Feature Array
    // access to the value with a pointer
    
    func extractFeatures(output features: MLMultiArray,name:String) -> [Prediction]{
        var classes = [Float](repeating: 0, count: 4)
        var predictions = [Prediction]()
        let numberOfGrids = features.shape[features.shape.count-1].intValue
        let gridSize = Yolov3.inputSize/Float(numberOfGrids)
        let numberOfClasses = 4
        let pointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
        if features.strides.count < 3 {
            print("out of bounds")
        }
        let channelStride = features.strides[features.strides.count-3].intValue
         let yStride = features.strides[features.strides.count-2].intValue
         let xStride = features.strides[features.strides.count-1].intValue
         
        func offset(ch: Int, x: Int, y: Int) -> Int {
           return ch * channelStride + y * yStride + x * xStride
         }

        for x in 0 ..< numberOfGrids {
          for y in 0 ..< numberOfGrids {
            for box_i in 0 ..< 3 {
              let boxOffset = box_i * (numberOfClasses + 5)
              let predX = Float(pointer[offset(ch: boxOffset, x: x, y: y)])
              let predY = Float(pointer[offset(ch: boxOffset + 1, x: x, y: y)])
              let predWidth = Float(pointer[offset(ch: boxOffset + 2, x: x, y: y)])
              let predheight = Float(pointer[offset(ch: boxOffset + 3, x: x, y: y)])
              let confidence = sigmoid(Float(pointer[offset(ch: boxOffset + 4, x: x, y: y)]))
                if confidence < 0.8 {
                continue
              }
                
            //the predX and predY are predict in relative to the grid cell.
            //to get the actual coords in the input picture
            //use sigmoid function to process predX/predY to get constraint value between 0 and 1
            //and then add the current x/y value in the loop
            //multiply by the gridSize
            //correct coordinate is obtained
                
            //EG predX is located in the middle of the input which is in grid 6 (0-12) and the gridSize is (512/13) = 39.38
            //so the sigmoid function will give out 0.5 then plus 6 and multiply by gridSize
            //the final X coordinate is (6 + 0.5) * 39.38 = 256
            //since the input size is 512 then half of 512 is 256
                
              let finalX = (sigmoid(predX) + Float(x)) * gridSize
              let finalY = (sigmoid(predY) + Float(y)) * gridSize
                
            //size of predWidth/predHeight are relative to the preset anchor box size
            //To get the actual width/height size need to multiply the preset anchor box size with the exponent of predWidth/predHeight
              let finalWidth = exp(predWidth) * self.anchors[name]![2 * box_i]
              let finalHeight = exp(predheight) * self.anchors[name]![2 * box_i + 1]
                
              for c in 0 ..< 4 {
                classes[c] = Float(pointer[offset(ch: boxOffset + 5 + c, x: x, y: y)])
              }
                
              classes = softmax(classes)
                
              let (classWithBestScore, bestScore) = argmax(classes)
                
              let confidenceInClass = bestScore * confidence
              if confidenceInClass > 0.7 {
                let prediction : Prediction = Prediction(classIndex: classWithBestScore,score: confidenceInClass,
                rect: CGRect(x: CGFloat(finalX - finalWidth / 2),
                             y: CGFloat(finalY - finalHeight / 2),
                             width: CGFloat(finalWidth),
                             height: CGFloat(finalHeight)))

                predictions.append(prediction)
              }
            }
          }
        }

        return predictions

    }
    
    //get the class index with the highest confidence score
    
    private func argmax(_ x: [Float]) -> (Int, Float) {
      let max = x.max()
      let index = x.firstIndex(of: max!)!
      return (index,max!)
    }
    
    //

    private func sigmoid(_ x: Float) -> Float {
      return 1 / (1 + exp(-x))
    }
    
    //transform array of float into array of probability

    private func softmax(_ x: [Float])->[Float] {
      let count = x.count
      var outputs = [Float](repeating: 0,
                            count: count)

      var inDescription = BNNSVectorDescriptor(size: count,
                                               data_type: .float)
      var outDescription = BNNSVectorDescriptor(size: count,
                                                data_type: .float)
      var activation = BNNSActivation(function: .softmax)
      var filterParameters = BNNSFilterParameters()

      let activationLayer = BNNSFilterCreateVectorActivationLayer(&inDescription,
                                                                  &outDescription,
                                                                  &activation,
                                                                  &filterParameters)

      BNNSFilterApply(activationLayer, x, &outputs)
      
      return outputs
    }
    
    //NMS is to prevent overlapped bounding box of the same class.
    //if two bounding boxes with same class overlap, if the IOU exceed threshold, only the higher score prediction will be displayed
    //remove the lower score prediction from the array
    
    private func nonMaxSuppression(boxes: inout [Prediction], threshold: Float) {
      var i = 0
      var j = 1
      while i < boxes.count {
         while j < boxes.count {
            if boxes[i].classIndex == boxes[j].classIndex {
                     if IOU(a: boxes[i].rect, b: boxes[j].rect) > threshold {
                          if boxes[i].score > boxes[j].score {
                                   boxes.remove(at: j)
                          }else{
                            boxes.remove(at: i)
                                          j = i + 1
                        }
                     }else{
                        j += 1
                }
            }else{
                j += 1
            }
        }
        i += 1
        j = i + 1
      }
    }

    //area of intersection / total area
    private func IOU(a: CGRect, b: CGRect) -> Float {
      
      let areaA = a.width * a.height
      let areaB = b.width * b.height
        
      if(areaA == 0.0 || areaB == 0.0){
            return 0.0
        }

      let intersection = a.intersection(b)
      let intersectionArea = intersection.width * intersection.height
      return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }


}


