import torch
import torch.nn as nn
from networks import Conv4, ResNet10, ResNet18

class SimpleShot(nn.Module):
  def __init__(
      self,
      input_dim : int = 84,
      num_classes : int = 100,
      network : str = "Conv-4",
  ):
    """
    Initialize a custom SimpleShot module.

    INPUT:
    - input_dim (int): dimension of image
    - num_classes (int): number of classes
    - network (str): type of convolutional network
    """
    super(SimpleShot, self).__init__()

    # convolutional network
    if network == "Conv-4":
      self.conv_net = Conv4()
    elif network == "ResNet-10":
      self.conv_net = ResNet10()
    elif network == "ResNet-18":
      self.conv_net = ResNet18()
    else:
      raise Exception("Inputted convolutional network not implemented by networks.py into SimpleShot.")

    # calculate feature dimension
    dummy_input = torch.zeros(1, 3, input_dim, input_dim)
    out = self.conv_net(dummy_input)
    self.feat_dim = out.view(1, -1).size(1)

    # linear classifier
    self.fc = nn.Linear(self.feat_dim, num_classes)
  
  def feature_extraction(self, x):
    x = self.conv_net(x)
    x = torch.flatten(x, start_dim=1)
    return x
  
  def forward(self, x):
    x = self.feature_extraction(x)
    x = self.fc(x)
    return x
  
  # @torch.no_grad()
  # def nearest_neighbor_classification(self, query_images, support_images, support_labels):
  #   # extract features fθ(I)
  #   query_features = self.feature_extraction(query_images)
  #   query_features = torch.flatten(query_features, start_dim=1)
    
  #   support_features = self.feature_extraction(support_images)
  #   support_features = torch.flatten(support_features, start_dim=1)

  #   # feature transformations (centering and l2)
  #   if self.support is not None:
  #     support_mean = torch.mean(self.feature_extraction(self.support), dim=0)
  #     support_mean = torch.flatten(support_mean)
  #     query_features = query_features - support_mean
  #     support_features = support_features - support_mean

  #   if self.l2norm:
  #     query_features = query_features / (1e-8 + torch.linalg.vector_norm(query_features, ord=2, dim=-1, keepdim = True))
  #     support_features = support_features / (1e-8 + torch.linalg.vector_norm(support_features, ord=2, dim=-1, keepdim=True))

  #   # nearest centroid approach (averaged feature vector for each class in support)
  #   support_classes = torch.unique(support_labels)
  #   class_centroids = []
  #   for c in support_classes:
  #     mask = (support_labels == c)
  #     class_features = support_features[mask]
  #     # averaged feature vector
  #     centroid = torch.mean(class_features, dim=0)
  #     class_centroids.append(centroid)

  #   # stack centroids --> single tensor
  #   centroids = torch.stack(class_centroids)

  #   number_queries = query_features.size(0)
  #   number_centroids = centroids.size(0)
  #   distances = torch.zeros(number_queries, number_centroids)

  #   for i in range(number_queries):
  #     for j in range(number_centroids):
  #       distances[i, j] = torch.sum((query_features[i]-centroids[j])**2)

  #   _, nearest_centroid = torch.min(distances, dim=1)
  #   predictions = support_classes[nearest_centroid]

  #   return predictions