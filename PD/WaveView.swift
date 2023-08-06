import SwiftUI
import PencilKit
import Alamofire

struct Wave: Shape {
    // allow SwiftUI to animate the wave phase
    var animatableData: Double {
        get { phase }
        set { self.phase = newValue }
    }
    
    // how high our waves should be
    var strength: Double
    
    // how frequent our waves should be
    var frequency: Double
    
    // how much to offset our waves horizontally
    var phase: Double
    
    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath()
        
        // calculate some important values up front
        let width = Double(rect.width)
        let height = Double(rect.height)
        let midWidth = width / 2
        let midHeight = height / 2
        let oneOverMidWidth = 1 / midWidth
        
        // split our total width up based on the frequency
        let wavelength = width / frequency
        
        // start at the left center
        //path.move(to: CGPoint(x: 0, y: midHeight))
        
        let graphWidth: CGFloat = 0.8  // Graph is 80% of the width of the view
        let amplitude: CGFloat = 0.25
        
        let origin = CGPoint(x: width * (1 - graphWidth) / 2, y: height * 0.50)
        
        
        path.move(to: origin)
        
        for angle in stride(from: 5.0, through: 360*frequency, by: 5.0) {
            let x = origin.x + CGFloat(angle/360.0) * wavelength
            let y = origin.y - CGFloat(sin(angle/180.0 * Double.pi)) * height * amplitude
            path.addLine(to: CGPoint(x: x, y: y))
        }
        
        
        
        return Path(path.cgPath)
    }
}

struct WaveView: View {
    @State private var phase = 0.0
    private var canvasView = PKCanvasView()
    @State private var test_result: String = ""
    
    struct Resp: Decodable, CustomStringConvertible {
        let result: String
        //let name: String
        
        
        var description: String {
            return "Resp: { result: \(result) }"
        }
    }
    
    var body: some View {
        VStack {
            Text("Spiral Test")
                .bold()
                .font(.title)
                .padding()
            ZStack() {
                Wave(strength: 50, frequency: 4, phase: self.phase)
                    .stroke(Color.blue, lineWidth: 5)
                MyCanvas(canvasView: canvasView)
                Text(test_result)
            }
            
            HStack(){
                Button("Clear", action: clear)
                Button("Save", action: saveImage)
                Button("Submit", action: analyzeImage)
            }
        }
        
        
    }
    
    func saveImage() {
        let image = canvasView.drawing.image(from: canvasView.drawing.bounds, scale: 1.0)
        UIImageWriteToSavedPhotosAlbum(image, self, nil, nil)
    }
    
    func analyzeImage() {
        let ts  = "2023"
        let place = "VA"
        
        let image = canvasView.drawing.image(from: canvasView.drawing.bounds, scale: 1.0).withTintColor(.systemPink, renderingMode: .alwaysOriginal)
        var parameters = ["ts":ts, "place":place]
        
        
        
        AF.upload(
            multipartFormData: { multipartFormData in
                if let spiral_data = image.pngData() {
                    multipartFormData.append(spiral_data, withName: "file", fileName: "spiral.png", mimeType: "image/png")
                }
                
                for (key, value) in parameters {
                    multipartFormData.append((value as! String).data(using: .utf8)!, withName: key)
                }
            },
            to: "https://qtechsolutions.net/pd/api/imageUpload", method: .post)
        .response { response in
            switch response.result {
            case .success(let data):
                let newJSONDecoder = JSONDecoder()
                if let result = try? newJSONDecoder.decode(Resp.self, from: data!){
                    test_result=result.result
                    print(result.result)
                    
                    
                }
            case .failure(let error):
                print(error)
            }
            
        }
        
        
        
        
    }
    
    func clear() {
        canvasView.drawing = PKDrawing()
        test_result=""
    }
}

struct WaveView_Previews: PreviewProvider {
    static var previews: some View {
        WaveView()
    }
}


