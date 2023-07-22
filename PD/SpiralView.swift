//
//  SpiralView.swift
//  PD
//
//  Created by ak on 6/24/23.
//


import SwiftUI
import Spiral
//import CompactSlider
import PencilKit

struct LineSpiralView: View {
    
    @State private var lineWidth: CGFloat = .lineWidth
    @State private var startAt: Double = 90
    @State private var endAt: Double = 1030
    @State private var smoothness: CGFloat = 50
    private var canvasView = PKCanvasView()
    
    var body: some View {
        
        VStack() {
            Text("Spiral Test")
                .bold()
                .font(.title)
                .padding()
            ZStack() {
                
                Spiral(
                    startAt: .degrees(startAt),
                    endAt: .degrees(endAt),
                    smoothness: smoothness
                )
                .stroke(
                    Color.blue,
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
            }
            //.padding()
            HStack(){
                Button("Clear", action: clear)
                Button("Submit", action: saveImage)
            }
        }
    }
    func saveImage() {
        let image = canvasView.drawing.image(from: canvasView.drawing.bounds, scale: 1.0)
        UIImageWriteToSavedPhotosAlbum(image, self, nil, nil)
    }
    
    func clear() {
        canvasView.drawing = PKDrawing()
    }
}
    
private extension CGFloat {
    static let lineWidth: CGFloat = 10
    
}

struct LineSpiralView_Previews: PreviewProvider {
    static var previews: some View {
        LineSpiralView()
    }
}


struct MyCanvas: UIViewRepresentable {
    var canvasView: PKCanvasView
    let picker = PKToolPicker.init()
    
    func makeUIView(context: Context) -> PKCanvasView {
        self.canvasView.tool = PKInkingTool(.pen, color: .black, width: 15)
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
