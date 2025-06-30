import Foundation
import Vision
import CoreImage
import CoreML

let fileManager = FileManager.default

class LandmarkExtractor {
    private let datasetPath = "./Face Shape Dataset Mix"
    private let outputPath = "./processed_landmarks_2"
    
    func processDataset() {
        print("Starting dataset processing...")
        
        // Create output directory
        createOutputDirectory()
        
        // Process training data
        print("Processing training data...")
        processImageSet(setType: "training_set")
        
        // Process testing data
        print("Processing testing data...")
        processImageSet(setType: "testing_set")
        
        print("Dataset processing completed!")
    }
    
    private func createOutputDirectory() {
        let fileManager = FileManager.default
        try? fileManager.createDirectory(atPath: outputPath, withIntermediateDirectories: true, attributes: nil)
    }
    
    private func processImageSet(setType: String) {
        let setPath = "\(datasetPath)/\(setType)"
        
        // Get all subdirectories inside the set (e.g., Heart, Square, etc.)
        guard let labelFolders = getSubdirectories(from: setPath) else {
            print("Failed to read subdirectories from \(setPath)")
            return
        }
        
        var csvData: [[String]] = []
        
        // CSV Header
        var header = ["image_name", "label"]
        
        // Add landmark coordinate columns (68 standard facial landmarks * 2 for x,y)
        for i in 0..<68 {
            header.append("landmark_\(i)_x")
            header.append("landmark_\(i)_y")
        }
        
        csvData.append(header)
        
        // Process each label folder (e.g., Heart, Square, etc.)
        var processedCount = 0
        
        for labelFolder in labelFolders {
            let imagesPath = "\(setPath)/\(labelFolder)"
            
            // Get all image files in the current label folder
            guard let imageFiles = getImageFiles(from: imagesPath) else {
                print("Failed to read images from \(imagesPath)")
                continue
            }
            
            print("Found \(imageFiles.count) images in \(labelFolder) folder")
            
            for imageFile in imageFiles {
                autoreleasepool {
                    let imagePath = "\(imagesPath)/\(imageFile)"
                    
                    // Extract landmarks
                    extractLandmarks(from: imagePath) { landmarks in
                        guard let landmarks = landmarks else {
                            print("Failed to extract landmarks from \(imageFile)")
                            return
                        }
                        
                        var row = [imageFile, labelFolder]
                        
                        // Add landmark coordinates (normalized to 0-1)
                        for landmark in landmarks.coordinates {
                            row.append(String(format: "%.6f", landmark.x))  // Format to 6 decimal places
                            row.append(String(format: "%.6f", landmark.y))  // Format to 6 decimal places
                        }
                        
                        // Add data to CSV
                        csvData.append(row)
                        processedCount += 1
                        
                        if processedCount % 100 == 0 {
                            print("Processed \(processedCount)/\(imageFiles.count) images...")
                        }
                    }
                }
            }
        }
        
        // Save CSV
        let csvContent = csvData.map { $0.joined(separator: ",") }.joined(separator: "\n")
        let csvFilePath = "\(outputPath)/\(setType)_landmarks.csv"
        
        do {
            try csvContent.write(toFile: csvFilePath, atomically: true, encoding: .utf8)
            print("Saved \(setType) landmarks to \(csvFilePath)")
            print("Successfully processed \(processedCount) images")
        } catch {
            print("Failed to save CSV: \(error)")
        }
    }
    
    private func getSubdirectories(from path: String) -> [String]? {
        let fileManager = FileManager.default
        do {
            let contents = try fileManager.contentsOfDirectory(atPath: path)
            return contents.filter { content in
                var isDir: ObjCBool = false
                let fullPath = "\(path)/\(content)"
                fileManager.fileExists(atPath: fullPath, isDirectory: &isDir)
                return isDir.boolValue
            }
        } catch {
            print("Error reading directory \(path): \(error)")
            return nil
        }
    }
    
    private func getImageFiles(from path: String) -> [String]? {
        let fileManager = FileManager.default
        do {
            let files = try fileManager.contentsOfDirectory(atPath: path)
            return files.filter { file in
                let lowercased = file.lowercased()
                return lowercased.hasSuffix(".jpg") || lowercased.hasSuffix(".jpeg") || lowercased.hasSuffix(".png")
            }
        } catch {
            print("Error reading directory \(path): \(error)")
            return nil
        }
    }
    
    private func extractLandmarks(from imagePath: String, completion: @escaping (FaceLandmarks?) -> Void) {
        // Load the image using CoreImage
        guard let ciImage = CIImage(contentsOf: URL(fileURLWithPath: imagePath)) else {
            print("Failed to load image at path: \(imagePath)")
            completion(nil)
            return
        }

        // Use Core Image to create a CGImage
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            print("Failed to convert CIImage to CGImage.")
            completion(nil)
            return
        }
        
        // Perform Vision for landmark detection
        let request = VNDetectFaceLandmarksRequest { [weak self] request, error in
            guard let self = self else {
                print("Self is nil")
                completion(nil)
                return
            }
            
            guard let results = request.results as? [VNFaceObservation], error == nil else {
                print("Error detecting faces: \(String(describing: error?.localizedDescription))")
                completion(nil)
                return
            }
            
            // Process the first detected face
            if let face = results.first {
                print("Detected face with bounding box: \(face.boundingBox)")
                
                // Now process the landmarks of this face
                if let landmarks = face.landmarks {
                    // Return the processed landmarks via the completion handler
                    completion(self.processFaceLandmarks(landmarks))
                } else {
                    completion(nil)
                }
            } else {
                completion(nil)
            }
        }

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Error performing Vision request: \(error.localizedDescription)")
            completion(nil)
        }
    }
    
    private func processFaceLandmarks(_ landmarks: VNFaceLandmarks2D) -> FaceLandmarks? {
        var coordinates: [CGPoint] = []

        // Access each landmark region and append points if available
        if let faceContour = landmarks.faceContour {
            addLandmarkPoints(faceContour, to: &coordinates)
        }
        if let leftEyebrow = landmarks.leftEyebrow {
            addLandmarkPoints(leftEyebrow, to: &coordinates)
        }
        if let rightEyebrow = landmarks.rightEyebrow {
            addLandmarkPoints(rightEyebrow, to: &coordinates)
        }
        if let nose = landmarks.nose {
            addLandmarkPoints(nose, to: &coordinates)
        }
        if let noseCrest = landmarks.noseCrest {
            addLandmarkPoints(noseCrest, to: &coordinates)
        }
        if let leftEye = landmarks.leftEye {
            addLandmarkPoints(leftEye, to: &coordinates)
        }
        if let rightEye = landmarks.rightEye {
            addLandmarkPoints(rightEye, to: &coordinates)
        }
        if let outerLips = landmarks.outerLips {
            addLandmarkPoints(outerLips, to: &coordinates)
        }
        if let innerLips = landmarks.innerLips {
            addLandmarkPoints(innerLips, to: &coordinates)
        }
        if let leftPupil = landmarks.leftPupil {
            addLandmarkPoints(leftPupil, to: &coordinates)
        }
        if let rightPupil = landmarks.rightPupil {
            addLandmarkPoints(rightPupil, to: &coordinates)
        }
        if let medianLine = landmarks.medianLine {
            addLandmarkPoints(medianLine, to: &coordinates)
        }

        // Ensure we have exactly 68 points (pad or truncate)
        while coordinates.count < 68 {
            coordinates.append(CGPoint(x: 0, y: 0))
        }
        if coordinates.count > 68 {
            coordinates = Array(coordinates.prefix(68))
        }

        return FaceLandmarks(coordinates: coordinates)
    }
    
    private func addLandmarkPoints(_ region: VNFaceLandmarkRegion2D?, to coordinates: inout [CGPoint]) {
        guard let region = region else { return }
        
        for point in region.normalizedPoints {
            coordinates.append(point)
        }
    }
}

struct FaceLandmarks {
    let coordinates: [CGPoint]
}

// Usage
let extractor = LandmarkExtractor()
extractor.processDataset()
