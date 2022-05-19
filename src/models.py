import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model, test_loader, args, device):
    model.to(device)
    #model = nn.DataParallel(model)
    model.eval()

    test_loss = 0.0
    correct = 0.0
    # criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            output = model(data)
            
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return test_loss,acc

    





    #     for batch_idx, (images, labels) in enumerate(test_loader):
    #         images, labels = images.to(device), labels.to(device)
    #         output_list, _ = model(images)

    #         ensemble_output = torch.stack(output_list, dim=2)
    #         ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_list)
            
    #         _, pred_labels_multi = torch.max(ensemble_output, 1)
    #         pred_labels_multi = pred_labels_multi.view(-1)
    #         correct_multi += torch.sum(torch.eq(pred_labels_multi, labels)).item()

    #         for i, single in enumerate(output_list):  
    #             _, pred_labels_single = torch.max(single, 1)
    #             pred_labels_single = pred_labels_single.view(-1)
    #             accuracy_single_list[i] += torch.sum(torch.eq(pred_labels_single, labels)).item()
                
    #         total += len(labels)

    #     accuracy_multi = correct_multi/total

    #     for i in range(len(accuracy_single_list)):
    #         accuracy_single_list[i] /= total
        
    # model.to(torch.device('cpu'))
    
    # return accuracy_multi, accuracy_single_list, loss


# if __name__ == "__main__":
#     print("Execute models.py")