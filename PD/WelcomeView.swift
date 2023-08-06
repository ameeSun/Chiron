//
//  WelcomeView.swift
//  PD
//
//  Created by lixun on 7/27/23.
//

import SwiftUI
import UIKit

struct WelcomeView: View {
    var body: some View {
        Text("Hello, World!")
    }
}

struct WelcomeView_Previews: PreviewProvider {
    static var previews: some View {
        WelcomeView()
    }
}

class ViewController1: UIViewController {
    
    @IBOutlet weak var backgroundGradientView: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
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
    }
    
    override var shouldAutorotate: Bool {
        return false
    }
}
