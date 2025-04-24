import torch
import torch.nn as nn

class SimpleShot(nn.Module):
  def __init__(
      self,
      input_dim : int = 84,
      hidden_dim : int = 64,
      num_classes : int = 100,
      l2norm : bool = True,
      support : torch.Tensor = None
  ):
    """
    Initialize a custom SimpleShot module.

    INPUT:
    - input_dim (int): dimension of image
    - hidden_dim (int): dimension of hidden layer
    - num_classes (int): number of classes
    - l2norm (bool): determine whether to use l2norm
    - support (torch.Tensor): support data for centering. If None, then no centering.
    """
    super(SimpleShot, self).__init__()

    # convolutional network
    self.conv_net = nn.Sequential(
      self.conv_block(3, hidden_dim),
      self.conv_block(hidden_dim, hidden_dim),
      self.conv_block(hidden_dim, hidden_dim),
      self.conv_block(hidden_dim, hidden_dim)
    )

    # calculate feature dimension
    dummy_input = torch.zeros(1, 3, input_dim, input_dim)
    out = self.conv_net(dummy_input)
    self.feat_dim = out.view(1, -1).size(1)

    # linear classifier
    self.fc = nn.Linear(self.feat_dim, num_classes)

    # l2norm and centering
    self.l2norm = l2norm
    self.support = support
    self.support_mean = None

    if self.support is not None:
      with torch.no_grad():
        support_features = self.feature_extraction(self.support)
        support_features = torch.flatten(support_features, start_dim=1)
        self.support_mean = torch.mean(support_features, dim=0)
  
  def conv_block(self, in_channels, out_channels):
    """
    Convolutional block. See link for implementation: https://arxiv.org/pdf/1703.05175#page=5.
    """
    model = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2)
    )
    return model
  
  def feature_extraction(self, x):
    return self.conv_net(x)
  
  def forward(self, x):
    x = self.feature_extraction(x)
    x = torch.flatten(x, start_dim=1)

    # TODO: implement centering
    if self.support_mean is not None:
      x = x - self.support_mean

    if self.l2norm:
      x = x / (1e-8 + torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim = True))

    x = self.fc(x)

    return x
  
  @torch.no_grad()
  def nearest_neighbor_classification(self, query_images, support_images, support_labels):
    # extract features fθ(I)
    query_features = self.feature_extraction(query_images)
    query_features = torch.flatten(query_features, start_dim=1)
    
    support_features = self.feature_extraction(support_images)
    support_features = torch.flatten(support_features, start_dim=1)

    # feature transformations (centering and l2)
    if self.support is not None:
      support_mean = torch.mean(self.feature_extraction(self.support), dim=0)
      support_mean = torch.flatten(support_mean)
      query_features = query_features - support_mean
      support_features = support_features - support_mean

    if self.l2norm:
      query_features = query_features / (1e-8 + torch.linalg.vector_norm(query_features, ord=2, dim=-1, keepdim = True))
      support_features = support_features / (1e-8 + torch.linalg.vector_norm(support_features, ord=2, dim=-1, keepdim=True))

    # nearest centroid approach (averaged feature vector for each class in support)
    support_classes = torch.unique(support_labels)
    class_centroids = []
    for c in support_classes:
      mask = (support_labels == c)
      class_features = support_features[mask]
      # averaged feature vector
      centroid = torch.mean(class_features, dim=0)
      class_centroids.append(centroid)

    # stack centroids --> single tensor
    centroids = torch.stack(class_centroids)

    number_queries = query_features.size(0)
    number_centroids = centroids.size(0)
    distances = torch.zeros(number_queries, number_centroids)

    for i in range(number_queries):
      for j in range(number_centroids):
        distances[i, j] = torch.sum((query_features[i]-centroids[j])**2)

    _, nearest_centroid = torch.min(distances, dim=1)
    predictions = support_classes[nearest_centroid]

    return predictions


