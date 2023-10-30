// The MIT License (MIT)
//
// Copyright (c) 2022 Alexey Bukhtin (github.com/buh).
//

import SwiftUI
import Spiral
import PencilKit
import Alamofire

struct PDSpiralView: View {
    
    @State private var lineWidth: CGFloat = .lineWidth
    @State private var startAt: Double = 90
    @State private var endAt: Double = 1030
    @State private var smoothness: CGFloat = 50
    private var canvasView = PKCanvasView()
    @State private var test_result: String = ""
    @State private var path = NavigationPath()
    
    
    struct Resp: Decodable, CustomStringConvertible {
        let result: String
        //let name: String
        
        
        var description: String {
            return "Resp: { result: \(result) }"
        }
    }
    
    var body: some View {
        
        VStack() {
            
            ZStack() {
                
                Spiral(
                    startAt: .degrees(startAt),
                    endAt: .degrees(endAt),
                    smoothness: smoothness
                )
                .stroke(
                    Color.pink,
                    style: .init(lineWidth: lineWidth, lineCap: .round, lineJoin: .round)
                )
                .opacity(0.5)
                .padding(lineWidth / 2)
                .padding(1)
                MyCanvas(canvasView: canvasView)
                
                /*
                 SpiralBaseControls(
                 startAt: $startAt,
                 endAt: $endAt,
                 smoothness: $smoothness
                 )*/
                
                Text(test_result)
                Image("favicon.ico")
                
            }
            VStack(){
                /*
                 Button("Redraw", action: clear)
                    .font(.title)
                 
                Button("Save", action: saveImage).font(/*@START_MENU_TOKEN@*/.title/*@END_MENU_TOKEN@*/)
                //Button("Stats", action: statsImage)
                //Button("Submit", action: analyzeImage)
                */
                HStack{
                    Button(action: clear
                    ) {
                        Image(systemName: "pencil")
                            .foregroundColor(.black)
                        Text("Redraw")
                            .padding(.horizontal)
                            .font(.headline)
                            .foregroundColor(.black)
                    }.padding().background(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(.pink, lineWidth: 1)
                    ).padding()
                    
                    Button(action: saveImage
                    ) {
                        Image(systemName: "square.and.arrow.up")
                            .foregroundColor(.black)
                        Text("Save")
                            .padding(.horizontal)
                            .font(.headline)
                            .foregroundColor(.black)
                    }.padding().background(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(.pink, lineWidth: 1)
                    ).padding()
                }
                NavigationLink("Submit") {
                    PDSpiralResultView(path: $path,canvasView: canvasView)
                }.navigationTitle("Spiral Test").font(.headline).padding().background(
                    RoundedRectangle(cornerRadius: 10)
                      .stroke(.pink, lineWidth: 1)
                ).padding().foregroundColor(.black)



            }
        }
    }
    func saveImage() {
        let image = canvasView.drawing.image(from: canvasView.drawing.bounds, scale: 1.0)
        UIImageWriteToSavedPhotosAlbum(image, self, nil, nil)
    }
    func statsImage() {
        let strokeCount = canvasView.drawing.strokes.count
        var count : Int=0
        for stroke in canvasView.drawing.strokes {
            let paths = stroke.path
            var path_starttime: Date = paths.creationDate
            
            for point in paths {
                count+=1
                var point_time=path_starttime + point.timeOffset
                test_result=test_result+"("+String(format: "%.1f",point.location.x)+","+String(format:"%.1f",point.location.y)+","+String(format: "%.3f",point_time.timeIntervalSince1970)+"),"
            }
        }
        test_result=String(count)
        
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

private extension CGFloat {
    static let lineWidth: CGFloat = 10
    
}

struct PDSpiralView_Previews: PreviewProvider {
    static var previews: some View {
        PDSpiralView()
    }
}


struct MyCanvas: UIViewRepresentable {
    var canvasView: PKCanvasView
    let picker = PKToolPicker.init()
    
    func makeUIView(context: Context) -> PKCanvasView {
        self.canvasView.tool = PKInkingTool(.pen, color: .systemPink, width: 20)
        self.canvasView.becomeFirstResponder()
        self.canvasView.backgroundColor = .clear
        self.canvasView.isOpaque = false
        self.canvasView.drawingPolicy = .anyInput
        return canvasView
    }
    
    func updateUIView(_ uiView: PKCanvasView, context: Context) {
        /*picker.addObserver(canvasView)
         picker.setVisible(true, forFirstResponder: uiView)
         DispatchQueue.main.async {
         uiView.becomeFirstResponder()
         }
         */
    }
}
