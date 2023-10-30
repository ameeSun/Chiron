//
//  SupportView.swift
//  PD
//
//  Created by ak on 6/20/23.
//

import SwiftUI
import WebKit

struct WebView : UIViewRepresentable {
    @State var url: String // 1
    
    func makeUIView(context: Context) -> some UIView {
        let webView = WKWebView(frame: .zero) // 2
        webView.load(URLRequest(url: URL(string: url)!)) // 3
        return webView
    }
    
    func updateUIView(_ uiView: UIViewType, context: Context) {} // 4
}


struct SupportView: View {
    @State var showWebView = false
    var body: some View {
        NavigationStack{
            Form{
                Image("pdsupport")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                Section{
                    Text("National Institute of Neurological Disorders and Stroke (NINDS)")
                        .font(.callout)
                    Link("braininfo@ninds.nih.gov", destination: URL(string: "mailto:braininfo@ninds.nih.gov")!)
                        .font(.callout)
                    Link("800-352-9424", destination: URL(string: "tel:8003529424")!)
                        .font(.callout)
                    Button {
                        showWebView = true
                    } label: {
                        Text("www.ninds.nih.gov")
                            .font(.callout)
                    }.sheet(isPresented: $showWebView) {
                        WebView(url: "https://www.google.com")
                    }
                }
                Section{
                    Text("National Institute of Environmental Health Sciences (NIEHS)")
                        .font(.callout)
                    Link("webcenter@niehs.nih.gov", destination: URL(string: "mailto:webcenter@niehs.nih.gov")!)
                        .font(.callout)
                    Link("919-541-3345", destination: URL(string: "tel:9195413345")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.niehs.nih.gov")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.niehs.nih.gov/health/topics/conditions/parkinson")
                    }
                }
                Section{
                    Text("American Parkinson Disease Association (APDA)")
                        .font(.callout)
                    Link("apda@apdaparkinson.org", destination: URL(string: "mailto:apda@apdaparkinson.org")!)
                        .font(.callout)
                    Link("800-223-2732", destination: URL(string: "tel:8002232732")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.apdaparkinson.org")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.apdaparkinson.org")
                    }
                }
                Section{
                    Text("Davis Phinney Foundation")
                        .font(.callout)
                    Link("info@davisphinneyfoundation.org", destination: URL(string: "mailto:info@davisphinneyfoundation.org")!)
                        .font(.callout)
                    Link("866-358-0285", destination: URL(string: "tel:8663580285")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.davisphinneyfoundation.org")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.davisphinneyfoundation.org")
                    }
                }
                Section{
                    Text("Michael J. Fox Foundation for Parkinson's Research")
                        .font(.callout)
                    Link("212-509-0995", destination: URL(string: "tel:2125090995")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.michaeljfox.org")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.michaeljfox.org")
                    }
                }
                Section{
                    Text("Parkinson Alliance")
                        .font(.callout)
                    Link("contact@parkinsonalliance.org", destination: URL(string: "mailto:contact@parkinsonalliance.org")!)
                        .font(.callout)
                    Link("800-579-8440", destination: URL(string: "tel:8005798440")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.parkinsonalliance.org")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.parkinsonalliance.org")
                    }
                }
                Section{
                    Text("Parkinson’s Resource Organization")
                        .font(.callout)
                    Link("info@parkinsonsresource.org", destination: URL(string: "mailto:info@parkinsonsresource.org")!)
                        .font(.callout)
                    Link("877-775-4111", destination: URL(string: "tel:8777754111")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.parkinsonsresource.org")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.parkinsonsresource.org")
                    }
                }
                Section{
                    Text("Parkinson's Foundation")
                        .font(.callout)
                    Link("helpline@parkinson.org", destination: URL(string: "mailto:helpline@parkinson.org")!)
                        .font(.callout)
                    Link("800-473-4636", destination: URL(string: "tel:8004734636")!)
                        .font(.callout)
                    Button {
                        self.showWebView = true
                    } label: {
                        Text("www.parkinson.org")
                            .font(.callout)
                    }.fullScreenCover(isPresented: $showWebView) {
                        WebView(url: "https://www.parkinson.org")
                    }
                }

            }
            .navigationBarTitle("Resources")
        }
        
    }
}
struct SupportView_Previews: PreviewProvider {
    static var previews: some View {
        SupportView()
    }
}
