//
//  HomeView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI
import PencilKit
import UIKit

struct nextButton: View {
    var body: some View {
        NavigationView{
            NavigationLink("Next", destination:  InfoView())
                .font(.system(.title2, design: .rounded, weight: .bold))
                .foregroundColor(.white)
                .fontWeight(.bold)
                .padding(20)
                .background(LinearGradient(gradient: Gradient(colors: [.pink, .purple]), startPoint: .leading, endPoint: .trailing).opacity(0.8))
                .cornerRadius(30)
                .background(Capsule().stroke(.white, lineWidth: 5))
                .shadow(color: .pink.opacity(0.5), radius: 2, x: 0, y: 2)
        }
        
    }
}
struct HomeView: View {
    private var change = false
    var body: some View {
        VStack {
            Spacer(minLength: 50)
            Text("Welcome!")
                .font(.custom("AmericanTypewriter", fixedSize: 36))
            Spacer(minLength: 50)
            Image("tulip")
                .clipShape(Circle())
                .overlay {
                    Circle().stroke(.gray, lineWidth: 2)
                }
            Spacer()
            Text("thisis isidih fhwef rehfh fhifhi fwdf hif fiuh sfuhi wfii efh f fiufhl iuf fhueifh")
                .padding(40)
            nextButton()
                
        }
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}
