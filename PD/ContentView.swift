//
//  ContentView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

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
