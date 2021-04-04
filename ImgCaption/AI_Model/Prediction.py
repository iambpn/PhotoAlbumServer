from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def greedyPrediction(model, word_to_idx, idx_to_word, image_feature, max_length, start='startseq', end='endseq',
                     logging=False):
    caption = start
    probabilities = [1]  # adding probability distribution of start to 1

    # iterate until max length
    for i in range(max_length):
        # encode word to index
        wordIndexes = [word_to_idx[word] for word in caption.split(' ') if word in word_to_idx]

        # add padding
        padded_wordIndexes = pad_sequences([wordIndexes], maxlen=max_length)

        # predict next word
        yhat = model.predict([image_feature, padded_wordIndexes], verbose=0)

        max_probability_index = np.argmax(yhat)  # argmax: Returns the index of the maximum values

        probabilities.append(yhat[0][max_probability_index])  # store maximum probability

        # map integer to word
        word = idx_to_word[max_probability_index]

        # append as input for generating the next word
        caption += ' ' + word

        # stop if we predict the end of the sequence
        if word == end:
            break

        if logging:
            print("... current caption length: %i words"
                  % (len(caption.split(' ')) - 1))

    # format probabilities
    probabilities = ['%.2f' % elem for elem in probabilities]

    return caption, probabilities


def beamSearchPrediction(model, word_to_idx, idx_to_word, image_feature, max_length, beam_width=3, startseq='startseq',
                    endseq='endseq'):

        start_list = [word_to_idx[startseq]]

        # result[0][0] = index of the starting word
        # result[0][1] = probability of the words predicted
        results = [[start_list, 0.0]]

        while len(results[0][0]) < max_length:
            temp = []
            for result in results:
                wordIndexes = result[0]

                padded_wordIndexes = pad_sequences([wordIndexes], maxlen=max_length)
                predictions = model.predict([image_feature, padded_wordIndexes], verbose=0)

                # Getting the top <beam_width>(n) predictions
                pred_sort_index = np.argsort(predictions[0])  # sort and return index of passed array asc -> dec
                top_pred_indexes = pred_sort_index[-beam_width:]  # slice last(greater) n beam width predictions

                # creating a new list so as to put them via the model again
                for wordIndex in top_pred_indexes:
                    next_cap, prob = result[0][:], result[1]  # extract prediction list and probability
                    next_cap.append(wordIndex)  # append predicted word index
                    prob += predictions[0][wordIndex]  # sum probability
                    temp.append([next_cap, prob])

            results = temp

            # Sorting according to the probabilities
            results = sorted(results, reverse=False, key=lambda l: l[1])

            # Getting the top n beam width words
            results = results[-beam_width:]

            # check for if all n words have endseq added
            # this is to reduce computation cost
            # when all n answers have endseq added then it means we do not need to iterate further
            # we can end the loop
            stop = []
            for result in results:
                if result[0][-1] == word_to_idx[endseq]:
                    stop.append(True)
                else:
                    stop.append(False)

            if all(stop):
                break

        results = results[-1][0]  # extract highest probability result
        intermediate_caption = [idx_to_word[i] for i in results]
        final_caption = []

        # beam search will run until the max sentence length so it will generate many endseq
        # this loop will clip sentence to 1st endseq
        for wordIndex in intermediate_caption:
            if wordIndex != endseq:
                final_caption.append(wordIndex)  # add word to final caption
            else:
                final_caption.append(wordIndex)
                break

        final_caption = ' '.join(final_caption)
        return final_caption