//
//  TestsView.swift
//  PD
//
//  Created by lixun on 8/6/23.
//

import SwiftUI
import PencilKit

struct TestsView: View {
     
    var body: some View {
        List {
            HStack(){
                Image(systemName: "circle.circle")
                    .foregroundColor(Color(.systemPink))
                NavigationLink("Spiral Test"){
                    PDSpiralView()
                }
                .padding()
            }
            HStack(){
                Image(systemName: "waveform.path")
                    .foregroundColor(Color(.systemPink))
                NavigationLink("Wave Test"){
                    WaveView()
                }
                .padding()
            }
        }
        .navigationTitle("Select Test to Perform")
    }
    
}

struct TestsView_Previews: PreviewProvider {
    
    static var previews: some View {
        @State  var path = NavigationPath()
        TestsView()
    }
}
