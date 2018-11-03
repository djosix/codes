from torch.autograd import Variable
import tqdm

from IPython import embed as debug

import config


def batch_variables(batch):
    '''
    Convert batch tensors into variables
    '''
    story, story_len = batch['story']
    query, query_len = batch['query']
    options, options_len = batch['options']
    answer = batch['answer']
    story = Variable(story)
    query = Variable(query)
    options = [Variable(option) for option in options]
    answer = Variable(answer)
    if config.cuda:
        story = story.cuda()
        query = query.cuda()
        options = [option.cuda() for option in options]
        answer = answer.cuda()
        story_len = story_len.cuda()
        query_len = query_len.cuda()
        options_len = [option_len.cuda()
                       for option_len in options_len]
    return (
        (story, story_len),
        (query, query_len),
        (options, options_len),
        answer
    )


def progress(desc, it):
    fmt = '{desc}: |{bar}|{percentage:3.0f}% {n_fmt}'
    return tqdm.tqdm(it, desc=desc, bar_format=fmt)
