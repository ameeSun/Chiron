//
//  InfoView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct InfoView: View {
    var body: some View {
        NavigationStack {
            List {
                HStack(){
                    Image(systemName: "person.fill.questionmark")
                        .foregroundColor(Color(.systemPink))
                    NavigationLink("What is Parkinson's?") {
                        WhatIsItView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "brain")
                        .foregroundColor(Color(.systemPink))
                    NavigationLink("Causes") {
                        CausesView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "stethoscope")
                        .foregroundColor(Color(.systemPink))
                    NavigationLink("Diagnosis") {
                        DiagnosisView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "person.fill.questionmark")
                        .foregroundColor(Color(.systemPink))
                    NavigationLink("Symptoms") {
                        SymptomsView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "person.fill.questionmark")
                        .foregroundColor(Color(.systemPink))
                    //Image("drugs")
                        //.resizable()
                        //.aspectRatio(contentMode: .fit)
                    NavigationLink("Treatment") {
                        TreatmentView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "person.fill.questionmark")
                        .foregroundColor(Color(.systemPink))
                    NavigationLink("Prevention") {
                        PreventionView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "person.fill.questionmark")
                        .foregroundColor(Color(.systemPink))
                    NavigationLink("Secondary Parkinsonism") {
                        SecondaryView()
                    }
                        .padding()
                }
            }
            .navigationTitle("Learn")
        }
    }
}


struct InfoView_Previews: PreviewProvider {
    static var previews: some View {
        InfoView()
    }
}
