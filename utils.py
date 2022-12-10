from torch.utils.data import DataLoader
from sklearn import model_selection
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.autograd import Variable
import math

class MaskBatch():
    "object for holding a batch of data with mask during training"

    def __init__(self, src, pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt, pad, device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask

def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1, size, size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa

def df_to_list(df,len_of_record):
    records = []
    df = np.array(df)
    num_batches = len(df)//len_of_record
    # for i in range(num_batches):
    #     records.append(df[(i*len_of_record):((i+1)*len_of_record)])
    for i in range(len(df) - len_of_record + 1):
        records.append(df[i:i + len_of_record])
    return(records)

def parse_datasets(device,batch_size,dataset,train_percentage=0.8):
    total_dataset = dataset

    # Shuffle and split
    if train_percentage > 0:
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size= train_percentage, shuffle = False)
    else:
        test_data = total_dataset

    if train_percentage > 0:
        train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
            collate_fn = lambda batch: variable_time_collate_fn(batch, device))
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    else:
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    if train_percentage > 0:
        data_objects = {"dataset_obj": total_dataset,
                        "train_dataloader": inf_generator(train_dataloader),
                        "test_dataloader": inf_generator(test_dataloader),
                        "n_train_batches": len(train_dataloader),
                        "n_test_batches": len(test_dataloader)}
    else:
        data_objects = {"dataset_obj": total_dataset,
                        "test_dataloader": inf_generator(test_dataloader),
                        "n_test_batches": len(test_dataloader)}
    return data_objects

def parse_datasets_separate(device,batch_size,train_dataset,val_dataset,test_dataset):

    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False,
        collate_fn = lambda batch: variable_time_collate_fn(batch, device))
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, device))
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: variable_time_collate_fn(batch, device))

    data_objects = {"train_dataloader": inf_generator(train_dataloader),
                    "val_dataloader": inf_generator(val_dataloader),
                    "test_dataloader": inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_val_batches": len(val_dataloader),
                    "n_test_batches": len(test_dataloader)}

    return data_objects

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def variable_time_collate_fn(batch, device=torch.device("cuda")):
    D = 4
    T = 50

    combined_tt = np.zeros([len(batch), T])
    combined_vol = np.zeros([len(batch), T])
    combined_price = np.zeros([len(batch), T])
    combined_event = np.zeros([len(batch), T])
    combined_state = np.zeros([len(batch), T])

    for i in range(len(batch)):
        combined_tt[i,:] = batch[i][:,0] / 10
        combined_event[i,:] = batch[i][:,1] - 1
        combined_vol[i,:] = batch[i][:,2]/100
        combined_state[i,:] = batch[i][:,-1]
        for j in range(T):
            if batch[i][j,1] == 1 or batch[i][j,1] == 2:
                combined_price[i,j] = max(((batch[i][j,6] - batch[i][j,3])/100 + 1),0)
            else:
                combined_price[i,j] = max(((batch[i][j,3] - batch[i][j,4])/100 + 1),0)

    data_dict = {
        "time_step": torch.Tensor(combined_tt).to(device),
        "volume": torch.Tensor(combined_vol).to(device),
        "price": torch.Tensor(combined_price).to(device),
        "event": torch.LongTensor(combined_event).to(device),
        "mkt_state": torch.Tensor(combined_state).to(device)}

    return data_dict

def get_next_batch(dataloader):
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()
    return data_dict

def predict_from_hidden(intens_at_samples, seq_dt, timestep, t_max, samples, batch_length, device='cuda'):

    intens_t_vals_sum = intens_at_samples.sum(dim=-1)
    integral_ = torch.cumsum(timestep * intens_t_vals_sum, dim=-1)
    # density for the time-until-next-event law
    density = intens_t_vals_sum * torch.exp(-integral_)
    taus = torch.tensor(
        np.linspace(0, 100 * t_max, 100 * samples, endpoint=False).astype(np.float32)).repeat(batch_length, 1).to(
        device)
    t_pit = taus * density  # integrand for the time estimator
    ratio = intens_at_samples / intens_t_vals_sum[:, :, None]
    prob_type = ratio * density[:, :, None]
    estimate_dt = (timestep * 0.5 * (t_pit[:, 1:] + t_pit[:, :-1])).sum(dim=-1)
    next_dt = seq_dt[:, -1]
    error_dt = torch.abs(torch.log10(estimate_dt) - torch.log10(next_dt)).mean()
    estimate_type_prob = (timestep * 0.5 * (prob_type[:, 1:, :] + prob_type[:, :-1, :])).sum(dim=1)
    next_pred_event_unknown_time = np.argmax(estimate_type_prob.cpu().numpy(), axis=-1)
    return next_pred_event_unknown_time, error_dt

def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def calculate_mid_price_array(df,only_pos=False):
    price_series = df.columns
    time_series = np.array(df.index)
    lob_array = np.array(df)

    first_positive = np.ones((lob_array.shape[0])) * -1
    last_negative = np.ones((lob_array.shape[0])) * -1

    for i in range(lob_array.shape[1]):
        j = lob_array.shape[1] - i - 1
        first_positive[lob_array[:, j] > 0] = j
        last_negative[lob_array[:, i] < 0] = i

    first_positive[first_positive==-1] = last_negative[first_positive==-1] + 1
    last_negative[last_negative==-1] = first_positive[last_negative==-1] - 1
    if only_pos:
        return first_positive, last_negative

    mid_price_array = np.zeros((lob_array.shape[0], 2))
    mid_price_array[:, 0] = time_series
    for i in range(lob_array.shape[0]):
        if last_negative[i] >= 0 and first_positive[i] >= 0:
            mid_price_array[i, 1] = (price_series[int(last_negative[i])] + price_series[int(first_positive[i])]) / 2
        else:
            mid_price_array[i, 1] = mid_price_array[i - 1, 1]
    return mid_price_array,first_positive,last_negative

def change_time_format(data):
    if isinstance(data,pd.DataFrame):
        data.index = data.index - pd.to_datetime('20200603')
        start_time = pd.to_timedelta('09:30:02')
        end_time = pd.to_timedelta('10:30:00')
        data = data[(data.index > start_time)&(data.index < end_time)]
        data.index = data.index / pd.Timedelta(1, unit='sec')
    else:
        raise Exception('type error')
    return data



def resample_regular(irregular_ts, interval):
    if isinstance(irregular_ts,pd.DataFrame):
        irregular_ts = irregular_ts.to_numpy()
    start = math.ceil(irregular_ts[0,0])
    end = irregular_ts[-1,0]
    num = int((end-start)/interval)
    end = start + num*interval
    time_sequence = np.linspace(start=start, stop=end, endpoint=True, num=num+1)
    price_sequence = np.zeros(np.shape(time_sequence))
    for i in range(len(time_sequence)):
        price_sequence[i] = irregular_ts[np.argmax(irregular_ts[:,0]/time_sequence[i]>=1),1]
    price_sequence[-1] = irregular_ts[-1,1]
    new_ts = np.concatenate((time_sequence[:,None],price_sequence[:,None]),axis=1)
    ret_ts = np.log(new_ts[1:,1]/new_ts[:-1,1])
    return new_ts, np.concatenate((time_sequence[1:,None],ret_ts[:,None]),axis=1)
