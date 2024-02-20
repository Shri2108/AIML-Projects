import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
import re

def preprocessor(text):
	text = re.sub('<[^>]*>', '', text) 
	text = re.sub('[\W]+', ' ', text.lower())
	return text

def classify_ticket(model,message):
  message = preprocessor(message)
  label = model.predict([message])[0]
  return {'Assigned Group:': label}

def main():
	st.write("""

	# Automated Support Ticket Assignment

	One of the key activities of any IT function is to “Keep the lights on” to ensure there is no impact to the Business operations. IT leverages Incident Management process to achieve the above Objective. An incident is something that is unplanned interruption to an IT service or reduction in the quality of an IT service that affects the Users and the Business. The main goal of Incident Management process is to provide a quick fix / workarounds or solutions that resolves the interruption and restores the service to its full capacity to ensure no business impact. In most of the organisations, incidents are created by various Business and IT Users, End Users/ Vendors if they have access to ticketing systems, and from the integrated monitoring systems and tools. Assigning the incidents to the appropriate person or unit in the support team has critical importance to provide improved user satisfaction while ensuring better allocation of support resources. The assignment of incidents to appropriate IT groups is still a manual process in many of the IT organisations. Manual assignment of incidents is time consuming and requires human efforts. There may be mistakes due to human errors and resource consumption is carried out ineffectively because of the misaddressing. On the other hand, manual assignment increases the response and resolution times which result in user satisfaction deterioration / poor customer service.

	This app predicts the assignment of support ticket to the appropriate group

	""")

	message_text = st.text_input("Input support ticket")
	model = joblib.load('trained_svc.pkl')
	
	if message_text != '':
		result = classify_ticket(model,message_text)
		st.write(result)

	# menu = ["Home","About"]
	# choice = st.sidebar.selectbox('Menu',menu)
	# if choice == 'Home':
	# 	st.subheader("Streamlit From Colab")	
if __name__ == '__main__':
	main()