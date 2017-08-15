// adapted from dascar
#pragma once

// OSG
#include <osgQt/GraphicsWindowQt>
#include <osgViewer/CompositeViewer>

// Qt
#include <QTimer>
#include <QGridLayout>

namespace app {
namespace viewer {

class ViewerWidget : public QWidget, public osgViewer::CompositeViewer
{
  public:
    ViewerWidget(QWidget* parent = 0,
                 Qt::WindowFlags f = 0,
                 osgViewer::ViewerBase::ThreadingModel tm =
                 osgViewer::CompositeViewer::SingleThreaded);

    QWidget* add_view_widget(osg::ref_ptr<osgQt::GraphicsWindowQt> gw);

    osg::ref_ptr<osgQt::GraphicsWindowQt> create_graphics_window(
        int x, int y, int w, int h,
        const std::string& name = "",
        bool window_decoration = false);

    virtual void paintEvent(QPaintEvent* event) { frame(); }

    osg::ref_ptr<osgViewer::View> GetView() { return _view; };

  private:
    // primary view
    osg::ref_ptr<osgViewer::View> _view;

  protected:

    ~ViewerWidget();

    QTimer _timer;
    QGridLayout* _grid;
};

} // namespace viewer
} // namespace app
