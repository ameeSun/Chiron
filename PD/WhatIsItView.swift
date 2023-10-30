//
//  WhatIsItView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct WhatIsItView: View {
    var body: some View {
            NavigationStack{
                Form{
                    Image("pdwhatis")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                    
                    Section{
                        Text("Parkinson’s disease is a progressive disorder that is caused by degeneration of nerve cells in the part of the brain called the substantia nigra, which controls movement. \n\nSymptoms usually begin gradually and worsen over time. As the disease progresses, people may have difficulty walking and talking. They may also have mental and behavioral changes, sleep problems, depression, memory difficulties, and fatigue.\n\nWhile virtually anyone could be at risk for developing Parkinson’s, some research studies suggest this disease affects more men than women. It’s unclear why, but studies are underway to understand factors that may increase a person’s risk. One clear risk is age: Although most people with Parkinson’s first develop the disease after age 60, about 5% to 10% experience onset before the age of 50. Early-onset forms of Parkinson’s are often, but not always, inherited, and some forms have been linked to specific alterations in genes. \n\nIt is estimated that 60,000 new cases of Parkinson’s disease are diagnosed each year, adding to the estimated one to 1.5 million Americans who currently have the disease. There were nearly 18,000 Parkinson’s disease-related deaths in the United States in 2003.")
                            .font(.callout)
                    }
                }
                .navigationBarTitle("What is Parkinson's")
            }
            
    }
}

struct WhatIsItView_Previews: PreviewProvider {
    static var previews: some View {
        WhatIsItView()
            .preferredColorScheme(.light)
    }
}
