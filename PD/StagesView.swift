//
//  StagesView.swift
//  PD
//
//  Created by lixun on 10/22/23.
//

import SwiftUI

struct StagesView: View {
    var body: some View {
        NavigationStack{
            Form{
                Image("pdscale")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Section{
                    Text("Parkinson’s disease can take years or even decades to cause severe effects. In 1967, two experts, Margaret Hoehn and Melvin Yahr, created the staging system for Parkinson’s disease. That staging system is no longer in widespread use because staging this condition is less helpful than determining how it affects each person’s life individually and then treating them accordingly.\n\nToday, the Movement Disorder Society-Unified Parkinson's Disease Rating Scale (MDS-UPDRS) is healthcare providers' main tool to classify this disease. The MDS-UPDRS examines four different areas of how Parkinson’s disease affects you:")
                        .font(.callout)
                }
                Section{
                    Text("Non-motor aspects of experiences of daily living. This section deals with non-motor (non-movement) symptoms like dementia, depression, anxiety and other mental ability- and mental health-related issues. It also asks questions about pain, constipation, incontinence, fatigue, etc.")
                        .font(.callout)
                }header: {
                    Text("Part 1:")
                }
                Section{
                    Text("Motor aspects of experiences of daily living. This section covers the effects on movement-related tasks and abilities. It includes your ability to speak, eat, chew and swallow, dress and bathe yourself if you have tremors and more.")
                        .font(.callout)
                }header: {
                    Text("Part 2:")
                }
                Section{
                    Text("Motor examination. A healthcare provider uses this section to determine the movement-related effects of Parkinson's disease. The criteria measure effects based on how you speak, facial expressions, stiffness and rigidity, walking gait and speed, balance, movement speed, tremors, etc.")
                        .font(.callout)
                }header: {
                    Text("Part 3:")
                }
                Section{
                    Text("Motor complications. This section involves a provider determining how much of an impact the symptoms of Parkinson’s disease are affecting your life. That includes both the amount of time you have certain symptoms each day, and whether or not those symptoms affect how you spend your time.")
                        .font(.callout)
                }header: {
                    Text("Part 4:")
                }
            }
            .navigationBarTitle("Disease Rating Scale")
        }
    }
}

struct StagesView_Previews: PreviewProvider {
    static var previews: some View {
        StagesView()
            .preferredColorScheme(.light)
    }
}
