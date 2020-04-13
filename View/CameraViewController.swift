//
//  CameraViewController.swift
//  fyp-yoloDetect
//
//  Created by Kee Heng on 31/03/2020.
//  Copyright © 2020 Kee Heng. All rights reserved.
//
//
import UIKit
import AVFoundation

class CameraViewController: UIViewController {

    let model = Yolov3()

    @IBOutlet weak var previewView: UIView!
    let session = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    var sessionPreviewLayer: AVCaptureVideoPreviewLayer?
    var boundingBoxLayer: BoundingBoxLayer!


    let cameraQueue = DispatchQueue(label: "yolov3.camera-queue")
    let semaphore = DispatchSemaphore(value: 1)

    override func viewDidLoad() {
        super.viewDidLoad()
        boundingBoxLayer = BoundingBoxLayer()
        boundingBoxLayer.previewFrame = previewView.frame

        newSession()
    }
    
    //start a new capture session
    //if sucessful, add the preview layer and bounding box layer to the view.
    //preview layer is used to display video as it’s captured by an input device
    //bounding box layer is used to display bounding boxes

    func newSession(){
        cameraQueue.async {
                  self.semaphore.wait()
                  let setUpSucess = self.isCameraSetUpSucess()
                  self.semaphore.signal()
                  DispatchQueue.main.async {
                    if setUpSucess {
                        if(self.sessionPreviewLayer != nil){
                            self.previewView.layer.addSublayer(self.sessionPreviewLayer!)
                            self.previewView.layer.addSublayer(self.boundingBoxLayer.layer)
                            self.sessionPreviewLayer?.frame = self.previewView.bounds
                        }
                    } else {
                      print("Fail to set up camera")
                    }
                  }
                }
    }


    override func viewDidDisappear(_ animated: Bool) {
      super.viewDidDisappear(animated)
         if session.isRunning {
           session.stopRunning()
         }
      semaphore.signal()
    }

    override func viewDidAppear(_ animated: Bool) {
      super.viewDidAppear(animated)
        if !session.isRunning {
            DispatchQueue.main.async {
              self.semaphore.wait()
              self.session.startRunning()
            }
          }
    }

    func isCameraSetUpSucess() -> Bool {

        var deviceInput: AVCaptureDeviceInput!

          let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first
          do {
              deviceInput = try AVCaptureDeviceInput(device: videoDevice!)
          } catch {
              print("Could not create video device input: \(error)")
              return false
          }


         session.beginConfiguration ()
         session.sessionPreset = .hd1280x720

        if session.canAddInput(deviceInput) {
         session.addInput(deviceInput)
        }

        if session.canAddOutput(videoOutput) {
        session.addOutput(videoOutput)

        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
      }

       let sessionPreviewLayer = AVCaptureVideoPreviewLayer(session: session)
       sessionPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
       sessionPreviewLayer.connection?.videoOrientation = .portrait
       self.sessionPreviewLayer = sessionPreviewLayer

       videoOutput.connection(with: AVMediaType.video)?.videoOrientation = .portrait
       session.commitConfiguration()
       return true
    }


}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {

  public func captureOutput(_ output: AVCaptureOutput,
                            didOutput sampleBuffer: CMSampleBuffer,
                            from connection: AVCaptureConnection) {


      if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            DispatchQueue.global().async {
                self.model.predict(pixelBuffer: imageBuffer)
                let predictions = self.model.predictions
                DispatchQueue.main.async {
                     if predictions != nil{
                        self.boundingBoxLayer.addBoundingBoxes(predictions: predictions!)
                        }
                }
            }


    }
  }


}


