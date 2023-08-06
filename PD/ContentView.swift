//
//  ContentView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI
import PencilKit

struct ContentView: View {
    var body: some View {
        TabView {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }.tag(0)
            InfoView()
                .tabItem {
                    Label("Information", systemImage: "books.vertical")
                }.tag(1)
            SupportView()
                .tabItem {
                    Label("Support", systemImage: "person.3")
                }
            
        }
        .accentColor(.red)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

class ViewController: UIViewController {
    
    let canvasView = PKCanvasView(frame: .zero)
    @IBOutlet weak var navigationBar: UINavigationBar!
    @IBOutlet weak var backgroundGradientView: UIView!

    override func viewDidLoad(){
        super.viewDidLoad()
        setNavigationBar()
        
        // Create a gradient layer.
        let gradientLayer = CAGradientLayer()
        // Set the size of the layer to be equal to size of the display.
        gradientLayer.frame = view.bounds
        // Set an array of Core Graphics colors (.cgColor) to create the gradient.
        // This example uses a Color Literal and a UIColor from RGB values.
        gradientLayer.colors = [#colorLiteral(red: 0, green: 0.5725490196, blue: 0.2705882353, alpha: 1).cgColor, UIColor(red: 252/255, green: 238/255, blue: 33/255, alpha: 1).cgColor]
        // Rasterize this static layer to improve app performance.
        gradientLayer.shouldRasterize = true
        // Apply the gradient to the backgroundGradientView.
        backgroundGradientView.layer.addSublayer(gradientLayer)
        
        canvasView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(canvasView)
        
        NSLayoutConstraint.activate([
            canvasView.topAnchor.constraint(equalTo: navigationBar.bottomAnchor),
            canvasView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            canvasView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            canvasView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        ])
    }
    override var shouldAutorotate: Bool {
            return false
        }
    
    func setNavigationBar() {
        if let navItem = navigationBar.topItem {
            let saveItem = UIBarButtonItem(title: "Save", style: .done, target: self, action: #selector(saveImage))
            let clearItem = UIBarButtonItem(title: "Clear", style: .plain, target: self, action: #selector(clear))
            let toggleItem = UIBarButtonItem(title: "Toggle Picker", style: .done, target: self, action: #selector(togglePicker))
                
            navItem.rightBarButtonItems?.append(clearItem)
            navItem.rightBarButtonItems?.append(toggleItem)
            navItem.rightBarButtonItems?.append(saveItem)
        }
    }
    
    @objc func saveImage() {
        let image = canvasView.drawing.image(from: canvasView.drawing.bounds, scale: 1.0)
        UIImageWriteToSavedPhotosAlbum(image, self, nil, nil)
    }
    
    @objc func clear() {
        canvasView.drawing = PKDrawing()
    }
    
    @objc func togglePicker() {
        if canvasView.isFirstResponder {
            canvasView.resignFirstResponder()
        }
        else {
            canvasView.becomeFirstResponder()
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        guard
            let window = view.window,
            let toolPicker = PKToolPicker.shared(for: window)
        else {
            return
        }
        
        toolPicker.setVisible(true, forFirstResponder: canvasView)
        toolPicker.addObserver(canvasView)
        canvasView.becomeFirstResponder()
    }
     
}
