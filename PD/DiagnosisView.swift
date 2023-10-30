//
//  DiagnosisView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct DiagnosisView: View {
    var body: some View {
        NavigationStack{
            Form{
                Image("pddiagnosis")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Section{
                    Text("There are currently no blood or laboratory tests to diagnose non-genetic cases of Parkinson’s. Doctors usually diagnose the disease by taking a person’s medical history and performing a neurological examination. If symptoms improve after starting to take medication, it’s another indicator that the person has Parkinson’s.\n\nA number of disorders can cause symptoms similar to those of Parkinson’s disease. People with Parkinson’s-like symptoms that result from other causes, such as multiple system atrophy and dementia with Lewy bodies, are sometimes said to have parkinsonism. While these disorders initially may be misdiagnosed as Parkinson’s, certain medical tests, as well as response to drug treatment, may help to better evaluate the cause. Many other diseases have similar features but require different treatments, so it is important to get an accurate diagnosis as soon as possible.\n\nThe mission of this app is to create a quick, accurate, and inexpensive method of diagnosing Parkinson’s.")
                        .font(.callout)
                }
            }
            .navigationBarTitle("Diagnosis")
        }
    }
}

struct DiagnosisView_Previews: PreviewProvider {
    static var previews: some View {
        DiagnosisView()
    }
}
