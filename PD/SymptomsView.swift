//
//  SymptomsView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct SymptomsView: View {
    var body: some View {
        NavigationStack{
            Form{
                Image("pdsymptoms")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Section{
                    
                    Text("The best-known symptoms of Parkinson's disease involve loss of muscle control. However, experts now know that muscle control-related issues aren't the only possible symptoms of Parkinson's disease.")
                        .font(.callout)
                    
                }
                Section{
                    let items: [String] = ["Slowed movements (bradykinesia). A Parkinson’s disease diagnosis requires that you have this symptom. People who have this describe it as muscle weakness, but it happens because of muscle control problems, and there’s no actual loss of strength.\n",
                                           "Tremor while muscles are at rest. This is a rhythmic shaking of muscles even when you’re not using them and happens in about 80% of Parkinson's disease cases. Resting tremors are different from essential tremors, which don’t usually happen when muscles are at rest.\n",
                                           "Rigidity or stiffness. Lead-pipe rigidity and cogwheel stiffness are common symptoms of Parkinson's disease. Lead-pipe rigidity is a constant, unchanging stiffness when moving a body part. Cogwheel stiffness happens when you combine tremor and lead-pipe rigidity. It gets its name because of the jerky, stop-and-go appearance of the movements (think of it as the second hand on a mechanical clock).\n",
                                           "Unstable posture or walking gait. The slowed movements and stiffness of Parkinson’s disease cause a hunched over or stooped stance. This usually appears as the disease gets worse. It’s visible when a person walks because they’ll use shorter, shuffling strides and move their arms less. Turning while walking may take several steps."]
                    let items1: [String] = ["Blinking less often than usual. This is also a symptom of reduced control of facial muscles.\n",
                                            "Cramped or small handwriting. Known as micrographia, this happens because of muscle control problems.\n",
                                            "Drooling. Another symptom that happens because of loss of facial muscle control.\n",
                                            "Mask-like facial expression. Known as hypomimia, this means facial expressions change very little or not at all.\n",
                                            "Trouble swallowing (dysphagia). This happens with reduced throat muscle control. It increases the risk of problems like pneumonia or choking.\n",
                                            "Unusually soft speaking voice (hypophonia). This happens because of reduced muscle control in the throat and chest."]
                    let list = items.toBulletList()
                    let list1 = items1.toBulletList()
                    
                    Text("Motor symptoms — which means movement-related symptoms — of Parkinson’s disease include the following:")
                        .font(.callout)
                        .fontWeight(.medium)
                    Text(list)
                        .font(.callout)
                    Text("Additional motor symptoms can include:")
                        .font(.callout)
                        .fontWeight(.medium)
                    Text(list1)
                        .font(.callout)
                }header: {
                    Text("Motor-related symptoms")
                }
                Section{
                    Text("Several symptoms are possible that aren't connected to movement and muscle control. In years past, experts believed non-motor symptoms were risk factors for this disease when seen before motor symptoms. However, there’s a growing amount of evidence that these symptoms can appear in the earliest stages of the disease. That means these symptoms might be warning signs that start years or even decades before motor symptoms.")
                        .font(.callout)
                    Text("Non-motor symptoms include:")
                        .font(.callout)
                        .fontWeight(.medium)
                    let items: [String] = ["Autonomic nervous system symptoms. These include orthostatic hypotension (low blood pressure when standing up), constipation and gastrointestinal problems, urinary incontinence and sexual dysfunctions.\n",
                                            "Depression\n",
                                            "Loss of sense of smell (anosmia)\n",
                                            "Sleep problems such as periodic limb movement disorder (PLMD), rapid eye movement (REM) behavior disorder and restless legs syndrome.\n",
                                            "Trouble thinking and focusing (Parkinson’s-related dementia)."]
                    let list = items.toBulletList()
                    Text(list)
                }header: {
                    Text("Non-motor symptoms")
                }
            }
            .navigationBarTitle("Symptoms")
        }
    }
}

struct SymptomsView_Previews: PreviewProvider {
    static var previews: some View {
        SymptomsView()
    }
}
extension Sequence where Self.Element == String {

  func toBulletList(_ bulletIndicator: String = "•",
                    itemSeparator: String = "\n",
                    spaceCount: Int = 2) -> String {
    let bullet = bulletIndicator + String(repeating: " ", count: spaceCount)
    let list = self
      .map { bullet + $0 }
      .reduce("", { $0 + ($0.isEmpty ? $0 : itemSeparator) + $1 })
    return list
  }
}
