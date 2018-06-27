from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def usingSilhouette(array, graphs):
    X = np.vstack(array)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    range_n_clusters = range(2, 8)
    clusters_dict = {}
    model_dic = {}
    hyper_params = {"n_init": [6, 8, 10, 12, 14],
                    "max_iter": [100, 200, 300, 400],
                    "tol": [1e-3, 1e-4, 1e-5]}

    for n_clusters in range_n_clusters:
        # Initialize the clusters with n_clusters value
        km_model = KMeans(n_clusters=n_clusters)
        ensemble = GridSearchCV(estimator= km_model, param_grid= hyper_params, cv= 5, n_jobs= -1)
        cluster_labels = ensemble.fit(X_scaled).predict(X_scaled)

        km_model = ensemble.best_estimator_
        model_dic[n_clusters] = km_model

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print("# Clusters: {}\n"
              "Best params: {}\n"
              "Silhouette avg: {}\n".format(n_clusters, ensemble.best_params_, silhouette_avg))
        clusters_dict[n_clusters] = silhouette_avg

        if graphs:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(13, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10])

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            # Labeling the clusters
            centers = km_model.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                        c="white", alpha=1, s=200, edgecolor='k')

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                            s=50, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')

            plt.show()

    clusters_dict = sorted(clusters_dict, key=clusters_dict.__getitem__, reverse= False)
    print("Best number of clusters: {}\n".format(clusters_dict[0:2]))
    # labels_list = {}
    # for i in range(2):
    #     labels_list[clusters_dict[i]] = dic_km_labels[clusters_dict[i]]

    best_models = {}
    for i in range(2):
        best_models[clusters_dict[i]] = model_dic[clusters_dict[i]]

    return clusters_dict[0:2], best_models


def kmEvaluation(km_model, data):
    X = data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cluster_labels = km_model.fit_predict(X_scaled)

    return cluster_labels