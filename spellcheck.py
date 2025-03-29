from textblob import TextBlob

incorrect_text = input("Enter the text: ")

print("Original: " + str(incorrect_text))

corrected_text = TextBlob(incorrect_text)

print("Corrected: " + str(corrected_text.correct()))
