import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """Pichai Sundararajan (born June 10, 1972), better known as Sundar Pichai (pronounced: /ˈsʊndɜːr pɪˈtʃeɪ/), is an Indian-American business executive.[3][4][5][6] He is the chief executive officer (CEO) of Alphabet Inc. and its subsidiary Google.[7]

    Pichai began his career as a materials engineer. Following a short stint at the management consulting firm McKinsey & Co., Pichai joined Google in 2004,[8] where he led the product management and innovation efforts for a suite of Google's client software products, including Google Chrome and ChromeOS, as well as being largely responsible for Google Drive. In addition, he went on to oversee the development of other applications such as Gmail and Google Maps.

    Pichai was selected to become the next CEO of Google on August 10, 2015, after previously being appointed chief product officer by then CEO Larry Page. On October 24, 2015, he stepped into the new position at the completion of the formation of Alphabet Inc., the new holding company for the Google company family. He was appointed to the Alphabet Board of Directors in 2017.[9] As of May 2025, his net worth is estimated at US$1.1 billion.[10]

    Early life and education
    Pichai was born on June 10, 1972[11][12][13] in Madurai, Tamil Nadu,[14][8][15] to a Tamil Hindu family. His mother, Lakshmi, was a stenographer, and his father, Regunatha Pichai, was an electrical engineer at GEC, the British conglomerate.[16][17]

    Pichai completed schooling in Jawahar Vidyalaya Senior Secondary School[18] in Ashok Nagar, Chennai and completed the Class XII from Vana Vani school at IIT Madras.[19][20] He earned a B.Tech in metallurgical engineering from IIT Kharagpur.[21] He holds an MS from Stanford University in materials science and engineering and an MBA from the Wharton School of the University of Pennsylvania,[22] where he was named a Siebel Scholar and a Palmer Scholar, respectively.[11][23][24]"""

    summary_template = """
    given hte information {information} about the person, I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
