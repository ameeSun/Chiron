//
//  SecondaryView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct SecondaryView: View {
    var body: some View {
        NavigationStack{
            Form{
                Image("pdsecondary")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Section{
                    Text("This is a disorder with symptoms similar to Parkinson's, but caused by medication side effects, different neurodegenerative disorders, illness or brain damage. As in Parkinson’s, many common symptoms may develop, including tremor; muscle rigidity or stiffness of the limbs; gradual loss of spontaneous movement, often leading to decreased mental skill or reaction time, voice changes, or decreased facial expression; gradual loss of automatic movement, often leading to decreased blinking, decreased frequency of swallowing, and drooling; a stooped, flexed posture with bending at the elbows, knees and hips; an unsteady walk or balance; and depression or dementia. Unlike Parkinson’s, the risk of developing secondary parkinsonism may be minimized by careful medication management, particularly limiting the usage of specific types of antipsychotic medications.\n\nMany of the medications used to treat this condition have potential side effects, so it is very important to work closely with the doctor on medication management. Unfortunately, secondary parkinsonism does not seem to respond as effectively to medical therapy as Parkinson's.")
                        .font(.callout)
                }
            }
            .navigationBarTitle("Secondary Parkinson's")
        }
    }
}

struct SecondaryView_Previews: PreviewProvider {
    static var previews: some View {
        SecondaryView()
    }
}
