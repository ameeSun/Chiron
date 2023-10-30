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
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("What is Parkinson's?") {
                        WhatIsItView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "brain")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Causes") {
                        CausesView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "stethoscope")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Diagnosis") {
                        DiagnosisView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "cross.case.fill")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Symptoms") {
                        //WaveView()
                        SymptomsView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "metronome.fill")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Disease Rating Scale") {
                        //WaveView()
                        StagesView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "pills.fill")
                        .foregroundColor(.pink.opacity(0.70))
                    //Image("drugs")
                        //.resizable()
                        //.aspectRatio(contentMode: .fit)
                    NavigationLink("Treatment") {
                        TreatmentView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "nosign")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Prevention") {
                        PreventionView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "cross.vial.fill")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Secondary Parkinsonism") {
                        SecondaryView()
                    }
                        .padding()
                }
                HStack(){
                    Image(systemName: "list.clipboard.fill")
                        .foregroundColor(.pink.opacity(0.70))
                    NavigationLink("Additional Resources") {
                        SupportView()
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
