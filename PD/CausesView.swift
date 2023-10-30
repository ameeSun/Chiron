//
//  CausesView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct CausesView: View {
    var body: some View {
        NavigationStack{
            Form{
                Image("pdcauses")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Section{
                    Text("The most prominent signs and symptoms of Parkinson’s disease occur when nerve cells in the basal ganglia, an area of the brain that controls movement, become impaired and/or die. Normally, these nerve cells, or neurons, produce an important brain chemical known as dopamine. When the neurons die or become impaired, they produce less dopamine, which causes the movement problems associated with the disease. Scientists still do not know what causes the neurons to die.\n\nPeople with Parkinson’s disease also lose the nerve endings that produce norepinephrine, the main chemical messenger of the sympathetic nervous system, which controls many functions of the body, such as heart rate and blood pressure. The loss of norepinephrine might help explain some of the non-movement features of Parkinson’s, such as fatigue, irregular blood pressure, decreased movement of food through the digestive tract, and sudden drop in blood pressure when a person stands up from a sitting or lying position.\n\nMany brain cells of people with Parkinson’s disease contain Lewy bodies, unusual clumps of the protein alpha-synuclein. Scientists are trying to better understand the normal and abnormal functions of alpha-synuclein and its relationship to genetic variants that impact Parkinson’s and Lewy body dementia.\n\nSome cases of Parkinson’s disease appear to be hereditary, and a few cases can be traced to specific genetic variants. While genetics is thought to play a role in Parkinson’s, in most cases the disease does not seem to run in families. Many researchers now believe that Parkinson’s results from a combination of genetic and environmental factors, such as exposure to toxins.")
                        .font(.callout)
                }
            }
            .navigationBarTitle("Causes of Parkinson's")
        }
    }
}

struct CausesView_Previews: PreviewProvider {
    static var previews: some View {
        CausesView()
    }
}
