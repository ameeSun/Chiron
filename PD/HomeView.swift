//
//  HomeView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI
import PencilKit

struct HomeView: View {
    @State private var path = NavigationPath()
    
    var body: some View {
        
            NavigationStack {
                VStack {
                    Spacer(minLength: 50)
                    Image("Chiron")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                    Text("Welcome!")
                        .font(.custom("AmericanTypewriter", fixedSize: 36))
                    Text("Click below to start diagnosis")
                        .padding(.bottom, 40.0)
                NavigationLink("Start Tests") {
                    TestsView()
                }.font(.system(.title2, design: .rounded, weight: .bold))
                        .foregroundColor(.white)
                        .fontWeight(.bold)
                        .padding(20)
                        .background(.pink.opacity(0.70))
                        .cornerRadius(30)
                        .background(Capsule().stroke(.white, lineWidth: 5))
                        .shadow(color: .pink.opacity(0.3), radius: 2, x: 0, y: 2)
                    Spacer(minLength: 50)
            }
                
        }
    }
}

struct HomeView_Previews: PreviewProvider {
    static var previews: some View {
        HomeView()
    }
}
