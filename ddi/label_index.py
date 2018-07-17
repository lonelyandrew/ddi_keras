#!/usr/bin/env python3

'''
This module provide a standard way to convert a string-type label of the
dataset into an integer index and vice versa.
At a lot of places, pre-training or training, we have to convert label and
index with each other, if there is not a standard conversion, it would be
very error-prone to map an icorrect index to a string label, and vice versa.
So we offer a standard way to fulfill this task, so do NOT convert labels and
indices by yourself, just call the given function.
'''

__class_list = ('negative', 'mechanism', 'effect', 'advise', 'int')


def label2ix(label):
    '''Convert a string label into an integer index.
    Args:
        label (str): The given label.
    Returns:
        Return an int index.
    '''
    return __class_list.index(label)


def ix2label(ix):
    '''Convert an integer index into a string label.
    Args:
        ix (int): The given index.
    Returns:
        Return a string label.
    '''
    return __class_list[ix]


def n_label():
    '''Get the count of the labels.
    Returns:
        Return an integer.
    '''
    return len(__class_list)


def labels():
    '''Get the lable list.
    Returns:
        Return a list of labels.
    '''
    return __class_list


if __name__ == '__main__':
    lable2ix('null')
    lable2ix('int')
    # lable2ix('fake_label')
    # lable2ix('')
    # lable2ix(0)
    # ix2label(-1)
    ix2label(0)
    # ix2label(10)
    # ix2label('null')
