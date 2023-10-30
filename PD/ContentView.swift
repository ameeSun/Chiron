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
        TabView () {
            HomeView()
                .tabItem {
                    Label("Home", systemImage: "house")
                }.tag(1)
            InfoView()
                .tabItem {
                    Label("Information", systemImage: "books.vertical")
                }.tag(2)
            SupportView()
                .tabItem {
                    Label("Support", systemImage: "person.3")
                }.tag(3)
            
        }
        .accentColor(.red)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

