//
//  PreventionView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI

struct PreventionView: View {
    var body: some View {
        NavigationStack{
            Form{
                Image("pdprevention")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                Section{
                    Text("Because the cause of Parkinson's is unknown, there are no proven ways to prevent the disease.\n\nSome research has shown that regular aerobic exercise might reduce the risk of Parkinson's disease.\n\nSome other research has shown that people who consume caffeine — which is found in coffee, tea and cola — get Parkinson's disease less often than those who don't drink it. Green tea also is related to a reduced risk of developing Parkinson's disease. However, it is still not known whether caffeine protects against getting Parkinson's or is related in some other way. Currently there is not enough evidence to suggest that drinking caffeinated beverages protects against Parkinson's.")
                        .font(.callout)
                }
            }
            .navigationBarTitle("Prevention")
        }
    }
}

struct PreventionView_Previews: PreviewProvider {
    static var previews: some View {
        PreventionView()
    }
}
